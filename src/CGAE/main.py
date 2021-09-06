import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.CGAE.model import train, impute, extract, MultiOmicsDataset
from src.util import logger
from sklearn.model_selection import StratifiedShuffleSplit


def run(args: dict) -> None:
    # Check cuda availability
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("Selected device: {}".format(device))
    torch.manual_seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], '{}'.format('CGAE'))
    os.makedirs(save_dir)

    # Load in data, depending on task
    # Task 1 : Imputation
    if args['task'] == 1:
        # variable y contains cancer type/cell type
        GE = np.load(args['data_path1'])
        ME = np.load(args['data_path2'])
        cancerType = np.load(args['cancer_type_index'])

        split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

        for trainValidInd, testInd in split1.split(GE, cancerType):

            # Get test split
            GEtest = GE[testInd]
            MEtest = ME[testInd]
            cancerTypetest = cancerType[testInd]

            # Get training and validation splits
            GEtrainValid = GE[trainValidInd]
            MEtrainValid = ME[trainValidInd]
            cancerTypetrainValid = cancerType[trainValidInd]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9)

        for trainInd, validInd in split2.split(GEtrainValid, cancerTypetrainValid):

            # Train splits
            GEtrain = GEtrainValid[trainInd]
            MEtrain = MEtrainValid[trainInd]
            cancerTypetrain = cancerTypetrainValid[trainInd]

            # Validation splits
            GEvalid = GEtrainValid[validInd]
            MEvalid = MEtrainValid[validInd]
            cancerTypevalid = cancerTypetrainValid[validInd]

    if args['task'] == 2:
        logger.success("Running Task 2: {} classification.".format(args['ctype']))
        # NOTE
        # For testing purposes, this code uses predefined splits, later this should be done everytime the model is run
        GEtrainctype = np.load(args['x_ctype_train_file'])
        GEtrainrest = np.load(args['x_train_file'])
        GEtrain = np.vstack((GEtrainctype, GEtrainrest))

        GEvalidctype = np.load(args['x_ctype_valid_file'])
        GEvalidrest = np.load(args['x_valid_file'])
        GEvalid = np.vstack((GEvalidctype, GEvalidrest))

        MEtrainctype = np.load(args['y_ctype_train_file'])
        MEtrainrest = np.load(args['y_train_file'])
        MEtrain = np.vstack((MEtrainctype, MEtrainrest))

        MEvalidctype = np.load(args['y_ctype_valid_file'])
        MEvalidrest = np.load(args['y_valid_file'])
        MEvalid = np.vstack((MEvalidctype, MEvalidrest))

    # Number of features
    input_dim1 = args['num_features']
    input_dim2 = args['num_features']

    encoder_layers = [args['latent_dim']]
    decoder_layers = [args['latent_dim']]

    # Initialize network model
    net = MultiOmicVAE(input_dim1, input_dim2, encoder_layers, decoder_layers, args['loss_function'], args['loss_function'],
                       args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['enc1_lr'],
                       args['dec1_lr'], args['enc1_last_activation'], args['enc1_output_scale'], args['enc2_lr'],
                       args['dec2_lr'], args['enc2_last_activation'], args['enc1_output_scale'], args['beta_start_value'],
                       args['zconstraintCoef'], args['crossPenaltyCoef']).to(device)

    net = net.double()

    logger.success("Initialized MultiOmicVAE model.")
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
    logger.info("Loading training and validation data into MultiOmicVAE...")

    dataTrain1 = torch.tensor(GEtrain, device=device)
    dataTrain2 = torch.tensor(MEtrain, device=device)

    dataValidation1 = torch.tensor(GEvalid, device=device)
    dataValidation2 = torch.tensor(MEvalid, device=device)

    datasetTrain = MultiOmicsDataset(dataTrain1, dataTrain2)
    datasetValidation = MultiOmicsDataset(dataValidation1, dataValidation2)

    train_loader = DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=True, num_workers=0,
                              drop_last=False)

    train_loader_eval = DataLoader(datasetTrain, batch_size=dataTrain1.shape[0], shuffle=False, num_workers=0,
                                   drop_last=False)

    valid_loader = DataLoader(datasetValidation, batch_size=dataValidation1.shape[0], shuffle=False, num_workers=0,
                              drop_last=False)

    # Training and validation

    train(device=device, net=net, num_epochs=args['epochs'], train_loader=train_loader,
          train_loader_eval=train_loader_eval, valid_loader=valid_loader,
          ckpt_dir=ckpt_dir, logs_dir=logs_dir, save_step=5, multimodal=True)


    # Extract Phase #

    # Imputation
    if args['task'] == 1:
        logger.info("Imputation: Extracting Z1 and Z2 using test set")

        dataExtract1 = GEtest
        dataExtract2 = MEtest

        dataExtract1 = torch.tensor(dataExtract1, device=device)
        dataExtract2 = torch.tensor(dataExtract2, device=device)

        datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2)

        extract_loader = DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False, num_workers=0,
                                    drop_last=False)

        # Compute imputation loss
        impute(net=net, model_file=os.path.join(ckpt_dir, "model_epoch{}.pth.tar".format(args['epochs'])), loader=extract_loader, save_dir=save_dir, multimodal=True)

    # Cancer stage prediction
    if args['task'] == 2:
        logger.info("Cancer Type Classification: Extracting Z1 and Z2 using {} set".format(args['ctype']))
        # Test sets are stratified data from cancer type into stages
        dataExtract1 = np.vstack((GEtrainctype, GEvalidctype, np.load(args['x_ctype_test_file'])))
        dataExtract2 = np.vstack((MEtrainctype, MEvalidctype, np.load(args['y_ctype_test_file'])))

        dataExtract1 = torch.tensor(dataExtract1, device=device)
        dataExtract2 = torch.tensor(dataExtract2, device=device)

        datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2)

        extract_loader = DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False, num_workers=0,
                                    drop_last=False)

        # Extract Z from all data from the chosen cancer type
        # Do predictions separately
        extract(net=net, model_file=os.path.join(ckpt_dir, "model_last.pth.tar".format(args['epochs'])), loader=extract_loader, save_dir=save_dir, multimodal=True)
