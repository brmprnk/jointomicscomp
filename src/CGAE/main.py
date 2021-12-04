import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.CGAE.model import train, impute, extract, MultiOmicsDataset
from src.util import logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter


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
        logger.info("Running Task {} on omic {} and omic {}".format(args['task'], args['data1'], args['data2']))

        # Load in data
        omic1 = np.load(args['data_path1'])
        omic2 = np.load(args['data_path2'])
        labels = np.load(args['labels'])
        labeltypes = np.load(args['labelnames'])

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

    if args['task'] == 2:
        logger.success("Running Task 2: {} classification.".format(args['ctype']))
        # NOTE
        # For testing purposes, this code uses predefined splits, later this should be done everytime the model is run
        Xtrainctype = np.load(args['x_ctype_train_file'])
        Xtrainrest = np.load(args['x_train_file'])
        Xtrain = np.vstack((Xtrainctype, Xtrainrest))

        Xvalctype = np.load(args['x_ctype_val_file'])
        Xvalrest = np.load(args['x_val_file'])
        Xval = np.vstack((Xvalctype, Xvalrest))

        Ytrainctype = np.load(args['y_ctype_train_file'])
        Ytrainrest = np.load(args['y_train_file'])
        Ytrain = np.vstack((Ytrainctype, Ytrainrest))

        Yvalctype = np.load(args['y_ctype_val_file'])
        Yvalrest = np.load(args['y_val_file'])
        Yval = np.vstack((Yvalctype, Yvalrest))

    # Number of features
    input_dim1 = args['num_features1']
    input_dim2 = args['num_features2']

    encoder_layers = [args['latent_dim']]
    decoder_layers = [args['latent_dim']]

    # Initialize network model
    net = CrossGeneratingVariationalAutoencoder(input_dim1, input_dim2, encoder_layers, decoder_layers, args['loss_function'],
                       args['loss_function'],
                       args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['enc1_lr'],
                       args['dec1_lr'], args['enc1_last_activation'], args['enc1_output_scale'], args['enc2_lr'],
                       args['dec2_lr'], args['enc2_last_activation'], args['enc1_output_scale'],
                       args['beta_start_value'],
                       args['zconstraintCoef'], args['crossPenaltyCoef']).to(device)

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

    train(device=device, net=net, num_epochs=args['epochs'], train_loader=train_loader,
          train_loader_eval=train_loader_eval, valid_loader=valid_loader,
          ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping, save_step=5, multimodal=True)

    # Extract Phase #

    # Imputation
    if args['task'] == 1:
        logger.info("Imputation: Extracting Z1 and Z2 using test set")

        dataExtract1 = omic1_test_file
        dataExtract2 = omic2_test_file

        dataExtract1 = torch.tensor(dataExtract1, device=device)
        dataExtract2 = torch.tensor(dataExtract2, device=device)

        datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2)

        extract_loader = DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False, num_workers=0,
                                    drop_last=False)

        # Compute imputation loss
        sample_names = np.load(args['sample_names']).astype(str)[test_ind]
        z1, z2 = impute(net=net,
                        model_file=ckpt_dir + '/model_last.pth.tar',
                        loader=extract_loader, save_dir=save_dir, sample_names=sample_names, multimodal=True)

        labels = np.load(args['labels']).astype(int)
        labeltypes = np.load(args['labelnames'])

        test_labels = labeltypes[[labels[test_ind]]]

        z1_plot = UMAPPlotter(z1, test_labels, "CGAE Z1: Task {} | {} & {} \n"
                                               "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                              .format(args['task'], args['data1'], args['data2'],
                                      args['epochs'], args['latent_dim'], args['enc1_lr'], args['batch_size']),
                              save_dir + "/{} UMAP_Z1.png".format('CGAE'))
        z1_plot.plot()

        z2_plot = UMAPPlotter(z2, test_labels, "CGAE Z2: Task {} | {} & {} \n"
                                               "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                              .format(args['task'], args['data1'], args['data2'],
                                      args['epochs'], args['latent_dim'], args['enc1_lr'], args['batch_size']),
                              save_dir + "/{} UMAP_Z2.png".format('CGAE'))
        z2_plot.plot()

    # Cancer stage prediction
    if args['task'] == 2:
        logger.info("Cancer Type Classification: Extracting Z1 and Z2 using {} set".format(args['ctype']))
        # Test sets are stratified data from cancer type into stages
        dataExtract1 = np.vstack((Xtrainctype, Xvalctype, np.load(args['x_ctype_test_file'])))
        dataExtract2 = np.vstack((Ytrainctype, Yvalctype, np.load(args['y_ctype_test_file'])))

        dataExtract1 = torch.tensor(dataExtract1, device=device)
        dataExtract2 = torch.tensor(dataExtract2, device=device)

        datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2)

        extract_loader = DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False, num_workers=0,
                                    drop_last=False)

        # Extract Z from all data from the chosen cancer type
        # Do predictions separately
        z1, z2 = extract(net=net, model_file=os.path.join(ckpt_dir, "model_last.pth.tar".format(args['epochs'])),
                         loader=extract_loader, save_dir=save_dir, multimodal=True)

        prediction_test_labels = np.load(args['ctype_test_file_labels'])

        z1_plot = UMAPPlotter(z1, prediction_test_labels, "CGAE Z1: Task {} on {} | {} & {} \n"
                                                          "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                              .format(args['task'], args['ctype'], args['data1'], args['data2'],
                                      args['epochs'], args['latent_dim'], args['lr'], args['batch_size']),
                              save_dir + "/{} UMAP.png".format('CGAE'))
        z1_plot.plot()

        z2_plot = UMAPPlotter(z2, prediction_test_labels, "CGAE Z2: Task {} on {} | {} & {} \n"
                                                          "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                              .format(args['task'], args['ctype'], args['data1'], args['data2'],
                                      args['epochs'], args['latent_dim'], args['lr'], args['batch_size']),
                              save_dir + "/{} UMAP.png".format('CGAE'))
        z2_plot.plot()
