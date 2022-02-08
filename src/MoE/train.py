import os
from datetime import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pickle

from src.MoE.model import MixtureOfExperts
import src.PoE.datasets as datasets
from src.PoE.evaluate import impute
from src.CGAE.model import train, MultiOmicsDataset
import src.util.logger as logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter
from src.util.evaluate import evaluate_imputation, save_factorizations_to_csv

import numpy as np
from sklearn.metrics import mean_squared_error


def loss_function(recon_omic1, omic1, recon_omic2, omic2, mu, log_var, kld_weight) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

    :return:
    """
    # Reconstruction loss
    recons_loss = 0
    if recon_omic1 is not None and omic1 is not None:
        recons_loss += F.mse_loss(recon_omic1, omic1)
    if recon_omic2 is not None and omic2 is not None:
        recons_loss += F.mse_loss(recon_omic2, omic2)

    recons_loss /= float(2)  # Account for number of modalities

    # KLD Loss
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    # Loss
    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}


def save_checkpoint(state, epoch, save_dir):
    """
    Saves a Pytorch model's state, and also saves it to a separate object if it is the best model (lowest loss) thus far

    @param state:       Python dictionary containing the model's state
    @param epoch:       Epoch number for save file name
    @param save_dir:      String of the folder to save the model to
    @return: None
    """
    # Save checkpoint
    torch.save(state, os.path.join(save_dir, 'trained_model_epoch{}.pth.tar'.format(epoch)))


# def train(args, model, train_loader, optimizer, epoch, tf_logger):
#     model.training = True
#     model = model.double()
#     model.train()
#
#     progress_bar = tqdm(total=len(train_loader))
#     train_loss_per_batch = np.zeros(len(train_loader))
#     train_recon_loss_per_batch = np.zeros(len(train_loader))
#     train_kl_loss_per_batch = np.zeros(len(train_loader))
#
#     # Incorporate MMVAE training function
#     b_loss = 0
#     for batch_idx, (omic1, omic2) in enumerate(train_loader):
#
#         if args['cuda']:
#             omic1 = omic1.cuda()
#             omic2 = omic2.cuda()
#
#         # refresh the optimizer
#         optimizer.zero_grad()
#
#         loss = -model.forward(omic1, omic2)
#         loss.backward()
#         optimizer.step()
#         b_loss += loss.item()
#
#         progress_bar.update()
#         train_loss_per_batch[batch_idx] = loss.item()
#
#
#
#     progress_bar.close()
#     if epoch % args['log_interval'] == 0:
#         tf_logger.add_scalar("train loss", train_loss_per_batch.mean(), epoch)
#         tf_logger.add_scalar("train reconstruction loss", train_recon_loss_per_batch.mean(), epoch)
#         tf_logger.add_scalar("train KL loss", train_kl_loss_per_batch.mean(), epoch)
#
#         print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_per_batch.mean()))
#         print('====> Epoch: {}\tReconstruction Loss: {:.4f}'.format(epoch, train_recon_loss_per_batch.mean()))
#         print('====> Epoch: {}\tKLD Loss: {:.4f}'.format(epoch, train_kl_loss_per_batch.mean()))


def test(args, model, val_loader, optimizer, epoch, tf_logger):
    model.training = False
    model.eval()
    validation_loss = 0
    validation_recon_loss = 0
    validation_kl_loss = 0


    for batch_idx, (omic1, omic2) in enumerate(val_loader):

        if args['cuda']:
            omic1 = omic1.cuda()
            omic2 = omic2.cuda()

        # for ease, only compute the joint loss in validation
        (joint_recon_omic1, joint_recon_omic2, joint_mu, joint_logvar) = model.forward(omic1, omic2)

        kld_weight = len(omic1) / len(val_loader.dataset)  # Account for the minibatch samples from the dataset

        # Compute joint loss
        joint_test_loss = loss_function(joint_recon_omic1, omic1,
                                        joint_recon_omic2, omic2,
                                        joint_mu, joint_logvar, kld_weight)

        validation_loss += joint_test_loss['loss']
        validation_recon_loss += joint_test_loss['Reconstruction_Loss']
        validation_kl_loss += joint_test_loss['KLD']

    validation_loss /= len(val_loader)
    validation_recon_loss /= len(val_loader)
    validation_kl_loss /= len(val_loader)

    if epoch % args['log_interval'] == 0:

        tf_logger.add_scalar("validation loss", validation_loss, epoch)
        tf_logger.add_scalar("validation reconstruction loss", validation_recon_loss, epoch)
        tf_logger.add_scalar("validation KL loss", validation_kl_loss, epoch)

        print('====> Epoch: {}\tValidation Loss: {:.4f}'.format(epoch, validation_loss))
        print('====> Epoch: {}\tReconstruction Loss: {:.4f}'.format(epoch, validation_recon_loss))
        print('====> Epoch: {}\tKLD Loss: {:.4f}'.format(epoch, validation_kl_loss))

    return validation_loss


def load_checkpoint(args, use_cuda=False):
    checkpoint = torch.load(args['pre_trained']) if use_cuda else \
        torch.load(args['pre_trained'], map_location=lambda storage, location: storage)

    trained_model = MVAE(args)
    trained_model.load_state_dict(checkpoint['state_dict'])
    return trained_model, checkpoint


def run(args) -> None:
    # # random seed
    # # https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.benchmark = True
    # torch.manual_seed(args['random_seed'])
    # np.random.seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], 'MoE')
    os.makedirs(save_dir)

    device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')
    # Define tensorboard logger
    # tf_logger = SummaryWriter(save_dir)

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

    dataTrain1 = torch.tensor(omic1_train_file, device=device)
    dataTrain2 = torch.tensor(omic2_train_file, device=device)

    dataValidation1 = torch.tensor(omic1_val_file, device=device)
    dataValidation2 = torch.tensor(omic2_val_file, device=device)

    datasetTrain = MultiOmicsDataset(dataTrain1, dataTrain2)
    datasetValidation = MultiOmicsDataset(dataValidation1, dataValidation2)

    train_loader = torch.utils.data.DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=False)
    val_loader = torch.utils.data.DataLoader(datasetValidation, batch_size=dataValidation1.shape[0], shuffle=False, num_workers=0, drop_last=False)

    if args['train_loader_eval_batch_size'] > 0:
        trnEvalBatchSize = args['train_loader_eval_batch_size']
    else:
        trnEvalBatchSize = len(train_dataset)

    train_loader_eval = torch.utils.data.DataLoader(datasetTrain, batch_size=trnEvalBatchSize, shuffle=False)

    if args['pre_trained'] != "":
        logger.info("Using Pre-Trained MVAE found at {}".format(args['pre_trained']))

        save_dir = os.path.dirname(args['pre_trained'])

        print("-----   Loading Trained Model   -----")
        model, checkpoint = load_checkpoint(args, args['cuda'])
        model.eval()

    else:
        # Setup and log model


        # Number of features
        input_dim1 = args['num_features1']
        input_dim2 = args['num_features2']

        assert input_dim1 == omic1.shape[1]
        assert input_dim2 == omic2.shape[1]

        encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
        decoder_layers = encoder_layers[::-1][1:]

        model = MixtureOfExperts(input_dim1, input_dim2, encoder_layers, decoder_layers,
         args['loss_function'], args['loss_function'], args['use_batch_norm'],
         args['dropout_probability'], args['optimizer'], args['enc1_lr'], args['dec1_lr'],
         args['enc1_last_activation'], args['enc1_output_scale'],
         args['enc2_lr'], args['dec2_lr'], args['enc2_last_activation'],
         args['enc2_output_scale'], args['enc_distribution'], args['beta_start_value'], args['K'])

        model.double()
        if device == torch.device('cuda'):
            model.cuda()
        else:
            args['cuda'] = False

        # Log Data shape, input arguments and model
        model_file = open("{}/MoE_Model.txt".format(save_dir), "a")
        model_file.write("Running at {}\n".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
        model_file.write("Input shape 1 : {}, {}\n".format(len(train_loader.dataset), args['num_features1']))
        model_file.write("Input shape 2 : {}, {}\n".format(len(train_loader.dataset), args['num_features2']))
        model_file.write("Input args : {}\n".format(args))
        model_file.write("PoE Model : {}".format(model))
        model_file.close()

        # Setup early stopping, terminates training when validation loss does not improve for early_stopping_patience epochs
        early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

        cploss = train(device=device, net=model, num_epochs=args['epochs'], train_loader=train_loader,
              train_loader_eval=train_loader_eval, valid_loader=val_loader,
              ckpt_dir=save_dir, logs_dir=save_dir, early_stopping=early_stopping, save_step=args['log_save_interval'], multimodal=True)

        #
        # for epoch in range(1, args['epochs'] + 1):
        #     train(args, model, train_loader, optimizer, epoch, tf_logger)
        #     validation_loss = test(args, model, val_loader, optimizer, epoch, tf_logger)
        #
        #     # Save the last model
        #     if epoch == args['epochs'] or epoch % args['log_save_interval'] == 0:
        #         save_checkpoint({
        #             'state_dict': model.state_dict(),
        #             'best_loss': validation_loss,
        #             'latent_dim': args['latent_dim'],
        #             'epochs': args['epochs'],
        #             'lr': args['lr'],
        #             'batch_size': args['batch_size'],
        #             'use_mixture': args['mixture'],
        #             'optimizer': optimizer.state_dict(),
        #         }, epoch, save_dir)
        #
        #     early_stopping(validation_loss)
        #
        #     # Stop training when not improving
        #     if early_stopping.early_stop:
        #         logger.info('Early stopping training since loss did not improve for {} epochs.'
        #                     .format(args['early_stopping_patience']))
        #         args['epochs'] = epoch  # Update nr of epochs for plots
        #         break

    # Extract Phase #
    logger.success("Finished training MVAE model. Now calculating task results.")

    # Imputation
    if args['task'] == 1:
        # Correct me if I'm wrong, but we can just get the x1_cross_hat and x2_cross_hat
        # from the model.forward

        logger.info("Task 1 Imputation: Extracting Z using test set")

        dataTest1 = torch.tensor(omic1_test_file, device=device)
        dataTest2 = torch.tensor(omic2_test_file, device=device)


        # 1 batch (whole test set)
        impute_loader = torch.utils.data.DataLoader(impute_dataset, batch_size=len(impute_dataset), shuffle=False)

        omic1_from_joint, omic2_from_joint, \
        omic1_from_omic1, omic2_from_omic1, \
        omic1_from_omic2, omic2_from_omic2 = impute(model, impute_loader, use_cuda=args['cuda'])

        # Reconstruction losses
        omic1_joint_reconstruction_loss = evaluate_imputation(omic1_from_joint, impute_dataset.omic1_data, args['num_features1'], 'mse')
        omic1_reconstruction_loss = evaluate_imputation(omic1_from_omic1, impute_dataset.omic1_data, args['num_features1'], 'mse')

        omic2_joint_reconstruction_loss = evaluate_imputation(omic2_from_joint, impute_dataset.omic2_data, args['num_features2'], 'mse')
        omic2_reconstruction_loss = evaluate_imputation(omic2_from_omic2, impute_dataset.omic2_data, args['num_features2'], 'mse')
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data1'], "both omics", omic1_joint_reconstruction_loss))
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data1'], args['data1'], omic1_reconstruction_loss))
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data2'], "both omics", omic2_joint_reconstruction_loss))
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data2'], args['data2'], omic2_reconstruction_loss))

        # Imputation losses
        NR_MODALITIES = 2

        # mse[i,j]: performance of using modality i to predict modality j
        mse = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
        rsquared = np.eye(NR_MODALITIES)
        spearman = np.zeros((NR_MODALITIES, NR_MODALITIES, 2), float) # ,2 since we report mean and median
        spearman_p = np.zeros((NR_MODALITIES, NR_MODALITIES), float)

        # From x to y
        mse[0, 1], rsquared[0, 1], spearman[0, 1], spearman_p[0, 1] =\
            evaluate_imputation(impute_dataset.omic2_data, omic2_from_omic1, args['num_features2'], 'mse'),\
            evaluate_imputation(impute_dataset.omic2_data, omic2_from_omic1, args['num_features2'], 'rsquared'),\
            evaluate_imputation(impute_dataset.omic2_data, omic2_from_omic1, args['num_features2'], 'spearman_corr'), \
            evaluate_imputation(impute_dataset.omic2_data, omic2_from_omic1, args['num_features2'], 'spearman_p')
        mse[1, 0], rsquared[1, 0], spearman[1, 0], spearman_p[1, 0] = \
            evaluate_imputation(impute_dataset.omic1_data, omic1_from_omic2, args['num_features1'], 'mse'),\
            evaluate_imputation(impute_dataset.omic1_data, omic1_from_omic2, args['num_features1'], 'rsquared'),\
            evaluate_imputation(impute_dataset.omic1_data, omic1_from_omic2, args['num_features1'], 'spearman_corr'), \
            evaluate_imputation(impute_dataset.omic1_data, omic1_from_omic2, args['num_features1'], 'spearman_p')

        performance = {'mse': mse, 'rsquared': rsquared, 'spearman_corr': spearman, 'spearman_p': spearman_p}
        print(performance)
        with open(save_dir + "/MoE_task1_results.pkl", 'wb') as f:
            pickle.dump(performance, f)

        logger.info("Imputation loss for {} from {} : {}".
                    format(args['data1'], args['data2'], mse[0, 1]))
        logger.info("Imputation loss for {} from {} : {}".
                    format(args['data2'], args['data1'], mse[1, 0]))

        # Get embeddings for UMAP
        for omic1, omic2 in impute_loader:  # Runs once since there is 1 batch

            if args['cuda']:
                omic1 = omic1.cuda()
                omic2 = omic2.cuda()

            z = model.extract(omic1, omic2)

            if args['cuda']:
                z = z.detach().cpu().numpy()
            else:
                z = z.detach().numpy()

            np.save("{}/task1_z.npy".format(save_dir), z)
            sample_names = np.load(args['sample_names'], allow_pickle=True).astype(str)
            save_factorizations_to_csv(z, sample_names[tcga_data.get_data_splits('test')], save_dir, 'task1_z')

            labels, label_types, test_ind = tcga_data.get_labels_partition("test")

            labels = labels[test_ind].astype(int)
            sample_labels = label_types[[labels]]

            # plot = UMAPPlotter(z, sample_labels, "{}: Task {} | {} & {} \n"
            #                                      "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
            #                    .format('MoE' if args['mixture'] else 'PoE', args['task'], args['data1'], args['data2'],
            #                            29, args['latent_dim'], args['lr'], args['batch_size']),
            #                    save_dir + "/{} UMAP.png".format('MoE' if args['mixture'] else 'PoE', 'MoE' if args['mixture'] else 'PoE'))
            #
            # plot.plot()

    if args['task'] == 2:
        print(model)
        logger.success("Extract z1 and z2 for classification of {}".format(args['ctype']))
        # Test sets are stratified data from cancer type into stages
        GEtrainctype = np.load(args['x_ctype_train_file'])
        GEvalidctype = np.load(args['x_ctype_valid_file'])
        MEtrainctype = np.load(args['y_ctype_train_file'])
        MEvalidctype = np.load(args['y_ctype_valid_file'])

        dataExtract1 = np.float32(np.vstack((GEtrainctype, GEvalidctype, np.load(args['x_ctype_test_file']))))
        dataExtract2 = np.float32(np.vstack((MEtrainctype, MEvalidctype, np.load(args['y_ctype_test_file']))))

        datasetExtract = datasets.TCGADataset(dataExtract1, dataExtract2)

        extract_loader = torch.utils.data.DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False)

        for batch_idx, (GE, ME) in enumerate(extract_loader):
            z = model.extract(GE, ME)
            z = z.detach().numpy()

            np.save("{}/task2_z.npy".format(save_dir), z)

        # Extract Z from all data from the chosen cancer type
        # Do predictions separately
