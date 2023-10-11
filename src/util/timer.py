import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from src.nets import *
from src.CGAE.model import train, impute, extract, MultiOmicsDataset, evaluateUsingBatches, evaluatePerDatapoint, timeTraining
from src.util import logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter
from src.baseline.baseline import classification, classificationMLP
from src.MoE.model import MixtureOfExperts
import src.PoE.datasets as datasets
from src.PoE.model import PoE
from src.PoE.train import timeTrainingPoE
import pickle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import anndata as ada
from scipy.sparse import csr_matrix
import uniport as up


def run(args, model_cls, fraction=1.0, n_epochs=30):
    # Check cuda availability
    device = torch.device('cuda')
    torch.manual_seed(args['random_seed'])

    n_modalities = args['nomics']

    # Load in data
    omics = [np.load(args['data_path%d' % (i+1)]) for i in range(n_modalities)]

    # Use predefined split
    train_ind = np.load(args['train_ind'])
    omics_train = [omic[train_ind] for omic in omics]

    if fraction < 1.0:
        # subsample data to study effects of training set size
        N = int(np.round(fraction * train_ind.shape[0]))
        omics_train = [omic[:N] for omic in omics_train]

    print('N = %d' % omics_train[0].shape[0])


    # Number of features
    input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

    likelihoods = [args['likelihood%d' % (i+1)] for i in range(n_modalities)]

    encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
    decoder_layers = encoder_layers[::-1][1:]

    # Initialize network model
    log_inputs = True

    if model_cls == 'cgae':
        net = CrossGeneratingVariationalAutoencoder(input_dims, encoder_layers, decoder_layers, likelihoods,
                           args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['lr'],  args['lr'],
                           args['enc_distribution'], args['beta_start_value'], args['zconstraintCoef'], args['crossPenaltyCoef'], log_input=log_inputs)

    elif model_cls == 'cvae':
        net = ConcatenatedVariationalAutoencoder(input_dims, encoder_layers, decoder_layers, likelihoods,
                           args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['lr'],  args['lr'],
                           args['enc_distribution'], args['beta_start_value'], log_input=log_inputs)

    elif model_cls == 'moe':
        llikScales = [1.0, 1.0]
        net = MixtureOfExperts(input_dims, encoder_layers, decoder_layers, likelihoods, args['use_batch_norm'],
                               args['dropout_probability'], args['optimizer'], args['lr'], args['lr'],
                               args['enc_distribution'], args['beta_start_value'], args['K'], llikScales, [-1, -1], log_inputs)
    else:
        raise NotImplementedError


    net = net.double()


    #dataTrain = [torch.tensor(omic, device=device) for omic in omics_train]
    dataTrain = [torch.tensor(omic) for omic in omics_train]

    datasetTrain = MultiOmicsDataset(dataTrain)

    train_loader = DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=True, num_workers=1, drop_last=False)
    # train_loader = DataLoader(datasetTrain, batch_size=32, shuffle=True, num_workers=1, drop_last=False)

    cgae_time = timeTraining(net, n_epochs, train_loader, device)

    return cgae_time



def runPoE(args, fraction=1.0, n_epochs=30):
    # Check cuda availability
    # Fetch Datasets
    save_dir = './PoE/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tcga_data = datasets.TCGAData(args, save_dir=save_dir)
    train_dataset = tcga_data.get_data_partition("train")

    device = torch.device('cuda')
    torch.manual_seed(args['random_seed'])

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


    model = PoE(args)
    model.cuda()
    model.double()

    # Preparation for training
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    n_modalities = args['nomics']

    # Load in data

    if fraction < 1.0:
        # subsample data to study effects of training set size
        N = int(np.round(fraction * train_dataset.omic2_data.shape[0]))
        #omics_train = [omic[:N] for omic in omics_train]
        train_dataset.omic2_data = train_dataset.omic2_data[:N]
        train_dataset.omic1_data = train_dataset.omic1_data[:N]


    print('N = %d' % train_dataset.omic2_data.shape[0])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    cgae_time = timeTrainingPoE(model, n_epochs, train_loader, optimizer)


    return cgae_time


def runUniport(args, fraction=1.0, n_epochs=30):
    # Check cuda availability
    # Fetch Datasets

    device = torch.device('cuda')
    torch.manual_seed(args['random_seed'])

    n_modalities = args['nomics']

    # Load in data
    unscaledomics = [np.load(args['data_path%d' % (i+1)]) for i in range(n_modalities)]
    omics = []
    scalers = []
    for i, omic1 in enumerate(unscaledomics):
        m = np.min(omic1)
        if m < 0:
            ss = MinMaxScaler().fit(omic1)
        else:
            ss = MaxAbsScaler().fit(omic1)

        omics.append(ss.transform(omic1))
        scalers.append(ss)

    # Use predefined split
    train_ind = np.load(args['train_ind'])
    omics_train = [omic[train_ind] for omic in omics]

    if fraction < 1.0:
        # subsample data to study effects of training set size
        N = int(np.round(fraction * train_ind.shape[0]))
        omics_train = [omic[:N] for omic in omics_train]

    print('N = %d' % omics_train[0].shape[0])

    # Load in data
    adtrain = []
    for i, omic in enumerate(omics_train):
        ds = ada.AnnData(csr_matrix(omic))
        ds.obs['source'] = args['data%d' % (i+1)]
        ds.obs['domain_id'] = 0
        ds.obs['domain_id'] = ds.obs['domain_id'].astype('category')
        adtrain.append(ds)

    Niter = int(np.round(n_epochs * omics_train[0].shape[0] / args['batch_size']))

    input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

    layers = args['latent_dim'].split('-')
    enc_arch = []
    for i in range(len(layers)-1):
        enc_arch.append(['fc', int(layers[i]), 1, 'relu'])

    enc_arch.append(['fc', int(layers[-1]), '', ''])

    net = up.Run(adatas=adtrain, mode='v', iteration=Niter, save_OT=True, out='latent', batch_size=64,
    lr=args['lr'], enc=enc_arch,
    gpu=0,
    loss_type='BCE',
    outdir='./up/',
    seed=124,
    num_workers=0,
    patience=100,
    batch_key='domain_id',
    source_name='source',
    model_info=False,
    verbose=False)



if __name__ == '__main__':
    import yaml
    Nepochs = 11

    #for fraction in [0.05, 0.1, 0.2, 0.5, 1.0]:
    for fraction in [0.1, 1.0, 0.5]:
        print('fraction: %f' % fraction, flush=True)

        # with open('configs/best-models/RNAADT_moe.yaml', 'r') as f:
        #     config = yaml.safe_load(f)
        # args = {**config['MoE'], **config['GLOBAL_PARAMS']}
        #

        #tt_moe = run(args, 'moe', fraction, Nepochs)
        #print(np.mean(tt_moe))
        #print(np.std(tt_moe), flush=True)

        #
        # with open('configs/best-models/RNAADT_cgae.yaml', 'r') as f:
        #     config = yaml.safe_load(f)
        # args = {**config['CGAE'], **config['GLOBAL_PARAMS']}
        #
        #
        # tt_cgae = run(args, 'cgae', fraction, Nepochs)
        # # print(np.mean(tt_cgae))
        # # print(np.std(tt_cgae), flush=True)
        # print(tt_cgae, flush=True)
        #
        # with open('configs/best-models/RNAADT_cvae.yaml', 'r') as f:
        #    config = yaml.safe_load(f)
        # args = {**config['CVAE'], **config['GLOBAL_PARAMS']}
        #
        #
        # tt_cvae = run(args, 'cvae', fraction, Nepochs)
        # print(np.mean(tt_cvae))
        # print(np.std(tt_cvae), flush=True)
        # print(tt_cvae, flush=True)
        # with open('configs/best-models/RNAADT_poe.yaml', 'r') as f:
        #     config = yaml.safe_load(f)
        # args = {**config['PoE'], **config['GLOBAL_PARAMS']}
        #
        #
        # tt_poe = runPoE(args, fraction, Nepochs)
        #
        # print(np.mean(tt_poe))
        # print(np.std(tt_poe), flush=True)

        with open('configs/best-models/RNAADT_uniport.yaml', 'r') as f:
            config = yaml.safe_load(f)
        args = {**config['UNIPORT'], **config['GLOBAL_PARAMS']}

        print('running uniport. Results printed on screen')
        runUniport(args, fraction, Nepochs)



        #with open('./timing-results-%s.pkl' % fraction, 'wb') as f:
        #    pickle.dump({'cgae': tt_cgae, 'cvae': tt_cvae, 'moe': tt_moe, 'poe': tt_poe}, f)
