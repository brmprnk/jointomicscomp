import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.PoE.model import PoE
import src.PoE.datasets as datasets
from src.PoE.evaluate import impute
import pickle
import sys
import yaml

config = sys.argv[1]
no = int(sys.argv[2])
Nperm = 10000

start = no * 400
end = start + 400

with open(config, 'r') as file:
    args = yaml.safe_load(file)


args = {**args['GLOBAL_PARAMS'], **args['PoE']}

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
if train_ind.shape[0] > 15000:
    train_ind = np.random.permutation(train_ind)[:15000]


omics_train = [omic[train_ind] for omic in omics]

# Number of features
input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

likelihoods = [args['likelihood%d' % (i+1)] for i in range(n_modalities)]

encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
decoder_layers = encoder_layers[::-1][1:]

# Initialize modelwork model
model = PoE(args)
if device == torch.device('cuda'):
    model.cuda()
else:
    args['cuda'] = False

model.double()

checkpoint = torch.load(args['pre_trained'])

model.load_state_dict(checkpoint['state_dict'])

# save_dir = os.path.join(args.results_path, 'results', '{}_{}_{}'
#                         .format(args['name'],
#                                 args['data1'],
#                                 args['data2']))
#
#save_dir = os.path.join(args['save_dir'], 'PoE/')

# Fetch Datasets
tcga_data = datasets.TCGAData(args, save_dir='./')
train_dataset = tcga_data.get_data_partition("train")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

model.eval()


kl1running = torch.zeros(model.latent_dim)
kl2running = torch.zeros(model.latent_dim)

N = omics_train[0].shape[0]


# calcualte MI for training data
for (b1, b2) in train_loader:
    b1 = b1.to(device)
    b2 = b2.to(device)

    with torch.no_grad():
        mu1, logvar1 = model.get_product_params(omic1=b1, omic2=None)
        mu2, logvar2 = model.get_product_params(omic1=None, omic2=b2)

        sigma1 = torch.exp(0.5 * logvar1)
        sigma2 = torch.exp(0.5 * logvar2)


    kl = -0.5 * torch.sum((1 + torch.log(sigma1 ** 2) - (mu1 ** 2) - (sigma1 ** 2)), 0)

    kl1running += kl.to('cpu')

    kl = -0.5 * torch.sum((1 + torch.log(sigma2 ** 2) - (mu2 ** 2) - (sigma2 ** 2)), 0)

    kl2running += kl.to('cpu')

miz_x1 = kl1running / N
miz_x2 = kl2running / N


# calcualte MI for permuted data#
kl1runningNull = torch.zeros(Nperm, model.latent_dim)
kl2runningNull = torch.zeros(Nperm, model.latent_dim)

Nfeats1 = omics[0].shape[1]
Nfeats2 = omics[1].shape[1]

for jj in range(Nperm):
    print(jj, flush=True)

    ii1 = torch.randperm(Nfeats1)
    ii2 = torch.randperm(Nfeats2)

    if jj < start:
        continue
    if jj == end:
        break

    for (b1, b2) in train_loader:
        b1 = b1.to(device)
        b2 = b2.to(device)

        b1 = b1[:, ii1]
        b2 = b2[:, ii2]

        with torch.no_grad():
            mu1, logvar1 = model.get_product_params(omic1=b1, omic2=None)
            mu2, logvar2 = model.get_product_params(omic1=None, omic2=b2)

            sigma1 = torch.exp(0.5 * logvar1)
            sigma2 = torch.exp(0.5 * logvar2)

        kl = -0.5 * torch.sum((1 + torch.log(sigma1 ** 2) - (mu1 ** 2) - (sigma1 ** 2)), 0)

        kl1runningNull[jj] += kl.to('cpu')

        kl = -0.5 * torch.sum((1 + torch.log(sigma2 ** 2) - (mu2 ** 2) - (sigma2 ** 2)), 0)

        kl2runningNull[jj] += kl.to('cpu')

# miz_x1_null = kl1runningNull / N
# miz_x2_null = kl2runningNull / N
#
# pvalue1 = torch.sum(miz_x1 < miz_x1_null, 0) / Nperm
# pvalue2 = torch.sum(miz_x2 < miz_x2_null, 0) / Nperm

resdict = {'x1stats': miz_x1, 'x2stats': miz_x2, 'kl1null': kl1runningNull, 'kl2null': kl2runningNull}

with open('results/mi/%s%s/PoE_%d.pkl' % (args['data1'], args['data2'], no), 'wb') as f:
    pickle.dump(resdict, f)
