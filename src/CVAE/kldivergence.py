import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.CGAE.model import train, impute, extract, MultiOmicsDataset, evaluateUsingBatches, evaluatePerDatapoint
import pickle
import sys
import yaml

config = sys.argv[1]

try:
    Nperm = int(sys.argv[2])
except IndexError:
    Nperm = 10000

with open(config, 'r') as file:
    args = yaml.safe_load(file)


args = {**args['GLOBAL_PARAMS'], **args['CVAE']}

# Check cuda availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(args['random_seed'])


n_modalities = args['nomics']

# Load in data
omics = [np.load(args['data_path%d' % (i+1)]) for i in range(n_modalities)]

labels = np.load(args['labels'])
labeltypes = np.load(args['labelnames'], allow_pickle=True)

# Use predefined split
train_ind = np.load(args['train_ind'])

# if training set is too big sub-sample to 20,000 data points to speed up computation
if train_ind.shape[0] > 20000:
    train_ind = np.random.permutation(train_ind)[:20000]

omics_train = [omic[train_ind] for omic in omics]

# Number of features
input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

likelihoods = [args['likelihood%d' % (i+1)] for i in range(n_modalities)]

encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
decoder_layers = encoder_layers[::-1][1:]

if args['data1'] == 'GE':
    log_inputs = [False, False]
else:
    if args['data2'] == 'ATAC':
        log_inputs = [True, False]
    else:
        log_inputs = [True, True]


# Initialize network model
if 'categorical' not in likelihoods:

    net = ConcatenatedVariationalAutoencoder(input_dims, encoder_layers, decoder_layers, likelihoods,
                       args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['lr'],  args['lr'],
                       args['enc_distribution'], args['beta_start_value'], log_input=log_inputs)

else:
    categories = [args['n_categories%d' % (i + 1)] for i in range(n_modalities)]
    net = ConcatenatedVariationalAutoencoder(input_dims, encoder_layers, decoder_layers, likelihoods,
                       args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['lr'],  args['lr'],
                       args['enc_distribution'], args['beta_start_value'], n_categories=categories)

net = net.double()

checkpoint = torch.load(args['pre_trained'])

net.encoders[0].load_state_dict(checkpoint['state_dict_enc'][0])
for i in range(n_modalities):
    net.decoders[i].load_state_dict(checkpoint['state_dict_dec'][i])

#net.load_state_dict(checkpoint['state_dict'])


dataTrain = [torch.tensor(omic1, device=device) for omic1 in omics_train]

datasetTrain = MultiOmicsDataset(dataTrain)

train_loader = DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=False, num_workers=0, drop_last=False)

net.eval()


kl1running = torch.zeros(net.z_dim)
kl2running = torch.zeros(net.z_dim)

N = omics_train[0].shape[0]

# calcualte MI for training data
for data in train_loader:
    batch = (data[0][0].double(), data[1][0].double())

    batch_x1 = (data[0][0].double(), torch.zeros_like(data[1][0]))
    batch_x2 = (torch.zeros_like(data[0][0]), data[1][0].double())

    #
    with torch.no_grad():
        qz_x1 = net.encoders[0](torch.cat(batch_x1, axis=1))
        qz_x2 = net.encoders[0](torch.cat(batch_x2, axis=1))

        mu1 = qz_x1.mean
        sigma1 = qz_x1.stddev

        mu2 = qz_x2.mean
        sigma2 = qz_x2.stddev


    kl = -0.5 * torch.sum((1 + torch.log(sigma1 ** 2) - (mu1 ** 2) - (sigma1 ** 2)), 0)

    kl1running += kl.to('cpu')

    kl = -0.5 * torch.sum((1 + torch.log(sigma2 ** 2) - (mu2 ** 2) - (sigma2 ** 2)), 0)

    kl2running += kl.to('cpu')

miz_x1 = kl1running / N
miz_x2 = kl2running / N


# calcualte MI for permuted data#
kl1runningNull = torch.zeros(Nperm, net.z_dim)
kl2runningNull = torch.zeros(Nperm, net.z_dim)

Nfeats1 = omics[0].shape[1]
Nfeats2 = omics[1].shape[1]

for jj in range(Nperm):
    print(jj)

    ii1 = torch.randperm(Nfeats1)
    ii2 = torch.randperm(Nfeats2)


    for data in train_loader:
        batch = (data[0][0].double(), data[1][0].double())

        x1 = batch[0][:, ii1]
        x2 = batch[1][:, ii2]

        batch_x1 = (x1, torch.zeros_like(x2))
        batch_x2 = (torch.zeros_like(x1), x2)


        with torch.no_grad():
            qz_x1 = net.encoders[0](torch.cat(batch_x1, axis=1))
            qz_x2 = net.encoders[0](torch.cat(batch_x2, axis=1))


            mu1 = qz_x1.mean
            sigma1 = qz_x1.stddev

            mu2 = qz_x2.mean
            sigma2 = qz_x2.stddev


        kl = -0.5 * torch.sum((1 + torch.log(sigma1 ** 2) - (mu1 ** 2) - (sigma1 ** 2)), 0)

        kl1runningNull[jj] += kl.to('cpu')

        kl = -0.5 * torch.sum((1 + torch.log(sigma2 ** 2) - (mu2 ** 2) - (sigma2 ** 2)), 0)

        kl2runningNull[jj] += kl.to('cpu')

miz_x1_null = kl1runningNull / N
miz_x2_null = kl2runningNull / N

pvalue1 = torch.sum(miz_x1 < miz_x1_null, 0) / Nperm
pvalue2 = torch.sum(miz_x2 < miz_x2_null, 0) / Nperm

resdict = {'x1stats': miz_x1, 'x2stats': miz_x2, 'x1null': miz_x1_null, 'x2null': miz_x2_null}

with open('results/mi/%s%s/CVAE.pkl' % (args['data1'], args['data2']), 'wb') as f:
    pickle.dump(resdict, f)
