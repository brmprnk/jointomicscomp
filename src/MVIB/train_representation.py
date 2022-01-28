import os

import numpy as np
import torch
from torch.utils.data import DataLoader


from src.nets import MVIB
from src.CGAE.model import train, extract, MultiOmicsDataset
from src.MVIB import training as training_module
from src.util import logger
from src.util.umapplotter import UMAPPlotter
from tensorboardX import SummaryWriter
from src.util.early_stopping import EarlyStopping
from src.util.evaluate import save_factorizations_to_csv
import pickle

def run(args: dict) -> None:
    logger.success("Now starting MVIB")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(args['random_seed'])

    # Create save directory
    save_dir = os.path.join(args['save_dir'], '{}'.format('MVIB'))
    os.makedirs(save_dir)


    # Load in data
    omic1 = np.load(args['data_path1'])
    omic2 = np.load(args['data_path2'])
    labels = np.load(args['labels'])
    labeltypes = np.load(args['labelnames'])

    assert omic1.shape[0] == omic2.shape[0]

    # Use predefined split
    train_ind = np.load(args['train_ind'])
    val_ind = np.load(args['val_ind'])
    test_ind = np.load(args['test_ind'])

    omic1_train_file = omic1[train_ind]
    omic1_val_file = omic1[val_ind]
    omic1_test_file = omic1[test_ind]
    omic2_train_file = omic2[train_ind]
    omic2_val_file = omic2[val_ind]
    omic2_test_file = omic2[test_ind]

    ytrain = labels[train_ind]
    yvalid = labels[val_ind]
    ytest = labels[test_ind]

    # Number of features
    input_dim1 = args['num_features1']
    input_dim2 = args['num_features2']

    assert input_dim1 == omic1.shape[1]
    assert input_dim2 == omic2.shape[1]


    encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
    mi_net_arch = [int(kk) for kk in args['mi_net_arch'].split('-')]

    net = MVIB(input_dim1, input_dim2, encoder_layers, mi_net_arch,

                       args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['enc1_lr'],
                       args['enc1_last_activation'], args['enc1_output_scale'], args['enc2_lr'],
                       args['enc2_last_activation'], args['enc2_output_scale'],
                       args['mi_net_lr'], args['beta']).to(device)
    net = net.double()

    logger.success("Initialized CrossGeneratingVariationalAutoencoder model.")
    logger.info(str(net))
    logger.info("Number of model parameters: ")
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("{}".format(num_params))

    # Create directories for checkpoint, sample and logs files
    ckpt_dir = save_dir + '/checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logs_dir = save_dir + '/logs'

    # Data loading
    logger.info("Loading training and validation data into CrossGeneratingVariationalAutoencoder...")

    dataTrain1 = torch.tensor(omic1_train_file, device=device)
    dataTrain2 = torch.tensor(omic2_train_file, device=device)

    dataValidation1 = torch.tensor(omic1_val_file, device=device)
    dataValidation2 = torch.tensor(omic2_val_file, device=device)

    datasetTrain = MultiOmicsDataset(dataTrain1, dataTrain2)
    datasetValidation = MultiOmicsDataset(dataValidation1, dataValidation2)

    train_loader = DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=True, num_workers=0,
                              drop_last=False)

    train_loader_eval = DataLoader(datasetTrain, batch_size=dataTrain1.shape[0], shuffle=False, num_workers=0,
                                   drop_last=False)

    valid_loader = DataLoader(datasetValidation, batch_size=dataValidation1.shape[0], shuffle=False, num_workers=0,
                              drop_last=False)

    # Setup early stopping, terminates training when validation loss does not improve for early_stopping_patience epochs
    early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

    # Training and validation

    cploss = train(device=device, net=net, num_epochs=args['epochs'], train_loader=train_loader,
          train_loader_eval=train_loader_eval, valid_loader=valid_loader,
          ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping, save_step=args['log_save_interval'], multimodal=True)

    # find best checkpoint based on the validation loss
    bestEpoch = args['log_save_interval'] * np.argmin(cploss)
    print(bestEpoch)

    if bestEpoch == 0:
        logger.info('WARNING!!! Training failed! Epoch 0 has the lowest loss.')

    logger.info("Using model from epoch %d" % bestEpoch)
    modelCheckpoint = ckpt_dir + '/model_epoch%d.pth.tar' % (bestEpoch)
    assert os.path.exists(modelCheckpoint), 'missing saved model epoch %d' % bestEpoch


    if args['task'] == 0:
        lossDict = {'epoch': bestEpoch, 'val_loss': np.min(cploss)}
        with open(save_dir + '/finalValidationLoss.pkl', 'wb') as f:
            pickle.dump(lossDict, f)


    # Load in data, depending on task

    # Extract Phase #
    assert args['task'] != 1

    # for TCGA we still need to figure out a solution here (maybe call this task 3 and have another if in which we just save the embeddings (or call r2py)?)
    if args['task'] == 2:
        logger.info("Classification")

        dataTest1 = torch.tensor(omic1_test_file, device=device)
        dataTest2 = torch.tensor(omic2_test_file, device=device)

        datasetTest = MultiOmicsDataset(dataTest1, dataTest2)


        train_loader = DataLoader(datasetTrain, batch_size=dataTrain1.shape[0], shuffle=False, num_workers=0,
                                       drop_last=False)
        valid_loader = DataLoader(datasetValidation, batch_size=dataValidation1.shape[0], shuffle=False, num_workers=0,
                                  drop_last=False)
        test_loader = DataLoader(datasetTest, batch_size=dataTest1.shape[0], shuffle=False, num_workers=0,
                          drop_last=False)


        z1train, z2train = extract(net, modelCheckpoint, train_loader, save_dir, multimodal=True)
        z1valid, z2valid = extract(net, modelCheckpoint, valid_loader, save_dir, multimodal=True)
        z1test, z2test = extract(net, modelCheckpoint, test_loader, save_dir, multimodal=True)

        logger.info("Using omic 1")
        predictions1, performance1 = classification(z1train, ytrain, z1valid, yvalid, z1test, ytest, np.array([1e-5, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])

        pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'pred': predictions1}

        logger.info("Using omic 2")
        predictions2, performance2 = classification(z2train, ytrain, z2valid, yvalid, z2test, ytest, np.array([1e-5, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])
        pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'pred': predictions2}


        logger.info("Saving results")
        with open(save_dir + "/CGAE_task2_results.pkl", 'wb') as f:
            pickle.dump({'omic1': pr1, 'omic2': pr2}, f)
