import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.CGAE.model import train, extract, MultiOmicsDataset
from src.util import logger
from sklearn.model_selection import train_test_split


def run(args: dict) -> None:
    # Check cuda availability
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("Selected device: {}".format(device))
    torch.manual_seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], '{}'.format('CGAE'))
    os.makedirs(save_dir)

    # Load in data
    GE = np.load(args['data_path1'])
    ME = np.load(args['data_path2'])
    input_dim1 = GE.shape[1]
    input_dim2 = ME.shape[1]
    dataTrain1, dataValidation1 = train_test_split(GE, test_size=args['validation_fraction'], shuffle=True,
                                                   random_state=args['random_seed'])
    dataTrain2, dataValidation2 = train_test_split(ME, test_size=args['validation_fraction'], shuffle=True,
                                                   random_state=args['random_seed'])

    encoder_layers = [25 ph6]
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

    dataTrain1 = torch.tensor(GE, device=device)
    dataTrain2 = torch.tensor(ME, device=device)

    dataValidation1 = torch.tensor(dataValidation1, device=device)
    dataValidation2 = torch.tensor(dataValidation2, device=device)

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
    # Data loading
    print("\n[*] Loading test data")

    dataExtract1 = GE
    dataExtract2 = ME

    dataExtract1 = torch.tensor(dataExtract1, device=device)
    dataExtract2 = torch.tensor(dataExtract2, device=device)

    datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2)

    extract_loader = DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False, num_workers=0,
                                drop_last=False)

    # Embedding extractor
    extract(device=device, net=net, model_file=os.path.join(ckpt_dir, "model_epoch40.pth.tar"), loader=extract_loader, save_dir=save_dir, multimodal=True)
