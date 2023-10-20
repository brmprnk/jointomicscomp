import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.CGAE.model import train, impute, extract, MultiOmicsDataset, evaluateUsingBatches, evaluatePerDatapoint
import pickle
import sys
import yaml
from src.MoE.model import MixtureOfExperts


config = sys.argv[1]

try:
    Nperm = int(sys.argv[2])
except IndexError:
    Nperm = 10000

with open(config, 'r') as file:
    args = yaml.safe_load(file)


args = {**args['GLOBAL_PARAMS'], **args['MoE']}

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

# if training set is too big sub-sample to 15,000 data points to speed up computation
if train_ind.shape[0] > 15000:
    train_ind = np.random.permutation(train_ind)[:15000]


omics_train = [omic[train_ind] for omic in omics]

# Number of features
input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

likelihoods = [args['likelihood%d' % (i+1)] for i in range(n_modalities)]
llikScales = [args['llikescale%d' % (i+1)] for i in range(n_modalities)]

encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
decoder_layers = encoder_layers[::-1][1:]

# Initialize network model

if 'categorical' in likelihoods:
    categories = [args['n_categories%d' % (i + 1)] for i in range(n_modalities)]
else:
    categories = None

if args['data1'] == 'GE':
    log_inputs = [False, False]
else:
    if args['data2'] == 'ATAC':
        log_inputs = [True, False]
    else:
        log_inputs = [True, True]


net = MixtureOfExperts(input_dims, encoder_layers, decoder_layers, likelihoods, args['use_batch_norm'],  args['dropout_probability'], args['optimizer'], args['lr'], args['lr'],  args['enc_distribution'], args['beta_start_value'], args['K'], llikScales, categories, log_inputs)

net.double()
if device == torch.device('cuda'):
    net.cuda()
else:
    args['cuda'] = False


net = net.double()

checkpoint = torch.load(args['pre_trained'])

for i in range(n_modalities):
    #print(i)
    net.encoders[i].load_state_dict(checkpoint['state_dict_enc'][i])
    net.decoders[i].load_state_dict(checkpoint['state_dict_dec'][i])


dataTrain = [torch.tensor(omic1, device=device) for omic1 in omics_train]

datasetTrain = MultiOmicsDataset(dataTrain)

train_loader = DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=False, num_workers=0, drop_last=False)
sys.exit(0)
net.eval()


kl1running = torch.zeros(net.z_dim)
kl2running = torch.zeros(net.z_dim)

N = omics_train[0].shape[0]

# calcualte MI for training data
for data in train_loader:
    batch = (data[0][0].double(), data[1][0].double())

    with torch.no_grad():
        qz_x1 = net.encoders[0](batch[0])
        qz_x2 = net.encoders[1](batch[1])

        mu1 = qz_x1.mean
        sigma1 = qz_x1.stddev

        b1 = sigma1 / np.sqrt(2)

        mu2 = qz_x2.mean
        sigma2 = qz_x2.stddev

        b2 = sigma2 / np.sqrt(2)


    kl = -1 + torch.log(b1) + ((torch.exp(-torch.abs(mu1)) + torch.abs(mu1)) / b1)
    kl = torch.sum(kl, 0)

    kl1running += kl.to('cpu')

    kl = -1 + torch.log(b2) + ((torch.exp(-torch.abs(mu2)) + torch.abs(mu2)) / b2)
    kl = torch.sum(kl, 0)

    kl2running += kl.to('cpu')

miz_x1 = kl1running / N
miz_x2 = kl2running / N

# calcualte MI for permuted data#
kl1runningNull = torch.zeros(Nperm, net.z_dim)
kl2runningNull = torch.zeros(Nperm, net.z_dim)

Nfeats1 = omics[0].shape[1]
Nfeats2 = omics[1].shape[1]

for jj in range(Nperm):
    print(jj, flush=True)

    ii1 = torch.randperm(Nfeats1)
    ii2 = torch.randperm(Nfeats2)


    for data in train_loader:
        batch = (data[0][0].double(), data[1][0].double())

        x1 = batch[0][:, ii1]
        x2 = batch[1][:, ii2]

        with torch.no_grad():
            qz_x1 = net.encoders[0](x1)
            qz_x2 = net.encoders[1](x2)

            mu1 = qz_x1.mean
            sigma1 = qz_x1.stddev

            b1 = sigma1 / np.sqrt(2)


            mu2 = qz_x2.mean
            sigma2 = qz_x2.stddev

            b2 = sigma2 / np.sqrt(2)


        kl = -1 + torch.log(b1) + ((torch.exp(-torch.abs(mu1)) + torch.abs(mu1)) / b1)
        kl = torch.sum(kl, 0)

        kl1runningNull[jj] += kl.to('cpu')

        kl = -1 + torch.log(b2) + ((torch.exp(-torch.abs(mu2)) + torch.abs(mu2)) / b2)
        kl = torch.sum(kl, 0)

        kl2runningNull[jj] += kl.to('cpu')

miz_x1_null = kl1runningNull / N
miz_x2_null = kl2runningNull / N

pvalue1 = torch.sum(miz_x1 < miz_x1_null, 0) / Nperm
pvalue2 = torch.sum(miz_x2 < miz_x2_null, 0) / Nperm

resdict = {'x1stats': miz_x1, 'x2stats': miz_x2, 'x1null': miz_x1_null, 'x2null': miz_x2_null}

with open('results/mi/%s%s/MoE.pkl' % (args['data1'], args['data2']), 'wb') as f:
    pickle.dump(resdict, f)
