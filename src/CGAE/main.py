import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.CGAE.model import train, impute, MultiOmicsDataset
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

    input_dim1 = GE.shape[1]
    input_dim2 = ME.shape[1]

    encoder_layers = [256]
    decoder_layers = [256]

    # Initialize network model
    net = MultiOmicVAE(input_dim1, input_dim2, encoder_layers, decoder_layers, args['loss_function'], args['loss_function'],
                       args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['enc1_lr'],
                       args['dec1_lr'], args['enc1_last_activation'], args['enc1_output_scale'], args['enc2_lr'],
                       args['dec2_lr'], args['enc2_last_activation'], args['enc1_output_scale'], args['beta_start_value'],
                       args['zconstraintCoef'], args['crossPenaltyCoef']).to(device)

    net = net.double()

    logger.success("Initialized MultiOmicVAE model.")
    print(net)
    logger.info("Number of model parameters: ")
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # Create directories for checkpoint, sample and logs files
    ckpt_dir = save_dir + '/checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logs_dir = save_dir + '/logs'

    # Data loading
    print("[*] Loading training and validation data...")

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
        print("Imputation: Extracting Z1 and Z2 using test set")

        dataExtract1 = GEtest
        dataExtract2 = MEtest

        dataExtract1 = torch.tensor(dataExtract1, device=device)
        dataExtract2 = torch.tensor(dataExtract2, device=device)

        datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2)

        extract_loader = DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False, num_workers=0,
                                    drop_last=False)

        # Compute imputation loss
        impute(net=net, model_file=os.path.join(ckpt_dir, "model_epoch{}.pth.tar".format(args['epochs'])), loader=extract_loader, save_dir=save_dir, multimodal=True)
