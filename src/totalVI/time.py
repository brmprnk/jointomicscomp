from src.totalVI.preprocessCITE import loadCITE, maskRNA, maskProteins
import torch
import os
from scvi.model import TOTALVI
import numpy as np
import pickle
from torch.distributions import Normal, Independent, Bernoulli
import src.util.mydistributions as mydist
import yaml

def runTimer(args, fraction, Nepochs):
    # Check cuda availability
    device = torch.device('cuda')
    torch.manual_seed(args['random_seed'])

    n_modalities = args['nomics']
    assert n_modalities == 2

    # Load in data
    mudataTrain, mudataTest = loadCITE(save=False, dataPrefix=args['mudata_path'])

    mudataTrain = mudataTrain[mudataTrain.obs[mudataTrain.obs['rna:donor'] != 'P2'].index].copy()
    mudataTrain.update()

    N = mudataTrain.obs.shape[0]
    if fraction < 1.0:
        N = int(np.round(fraction * N))

        mudataTrain = mudataTrain[mudataTrain.obs.iloc[:N].index]
        mudataTrain.update()

    TOTALVI.setup_mudata(
        mudataTrain,
        rna_layer="counts",
        protein_layer=None,
        #batch_key="batch",
        modalities={
            "rna_layer": "rna_subset",
            "protein_layer": "protein",
            #"batch_key": "rna_subset",
        }
    )


    params = dict(n_latent=args['zdim'], latent_distribution=args['enc_distribution'], n_hidden=args['n_neurons_per_layer'], n_layers_encoder=args['n_layers'], n_layers_decoder=args['n_layers'])

    net = TOTALVI(mudataTrain, **params)


    trainSettings = {'max_epochs': Nepochs, 'lr': args['lr'], 'accelerator': 'gpu', 'devices': 'auto', 'train_size': 0.999, 'shuffle_set_split': False, 'batch_size': args['batch_size'], 'early_stopping': True, 'early_stopping_patience': 10, 'early_stopping_min_delta': 1., 'reduce_lr_on_plateau': False, 'n_steps_kl_warmup': 1, 'adversarial_classifier': False}
    net.train(**trainSettings)



if __name__ == '__main__':
    with open('configs/best-models/RNAADT_totalvi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    args = {**config['totalVI'], **config['GLOBAL_PARAMS']}

    Nepochs = 11

    for fraction in [0.05, 0.1, 0.2, 0.5, 1.0]:
        print(fraction)
        runTimer(args, fraction, Nepochs)
        # print(np.mean(tt_cgae))
        # print(np.std(tt_cgae), flush=True)


        print('\n\n')
