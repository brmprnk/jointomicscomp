import os
from datetime import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from src.MVAE.model import MVAE
import src.MVAE.datasets as datasets
from src.MVAE.evaluate import impute
import src.util.logger as logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter

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


def train(args, model, train_loader, optimizer, epoch, tf_logger):
    model.training = True
    model.train()

    progress_bar = tqdm(total=len(train_loader))
    train_loss_per_batch = np.zeros(len(train_loader))
    train_recon_loss_per_batch = np.zeros(len(train_loader))
    train_kl_loss_per_batch = np.zeros(len(train_loader))
    for batch_idx, (omic1, omic2) in enumerate(train_loader):
        # refresh the optimizer
        optimizer.zero_grad()

        kld_weight = len(omic1) / len(train_loader.dataset)  # Account for the minibatch samples from the dataset

        # compute reconstructions using all the modalities
        (joint_recon_omic1, joint_recon_omic2, joint_mu, joint_logvar) = model.forward(omic1, omic2)

        # compute reconstructions using each of the individual modalities
        (omic1_recon_ge, omic1_recon_me, omic1_mu, omic1_logvar) = model.forward(omic1=omic1)

        (omic2_recon_ge, omic2_recon_me, omic2_mu, omic2_logvar) = model.forward(omic2=omic2)

        # Compute joint train loss
        joint_train_loss = loss_function(joint_recon_omic1, omic1,
                                         joint_recon_omic2, omic2,
                                         joint_mu, joint_logvar, kld_weight)

        # compute loss with single modal inputs
        omic1_train_loss = loss_function(omic1_recon_ge, omic1,
                                      omic1_recon_me, omic2,
                                      omic1_mu, omic1_logvar, kld_weight)

        omic2_train_loss = loss_function(omic2_recon_ge, omic1,
                                      omic2_recon_me, omic2,
                                      omic2_mu, omic2_logvar, kld_weight)

        train_loss = joint_train_loss['loss'] + omic1_train_loss['loss'] + omic2_train_loss['loss']
        train_loss_per_batch[batch_idx] = train_loss
        train_recon_loss = joint_train_loss['Reconstruction_Loss'] + omic1_train_loss['Reconstruction_Loss'] + \
                           omic2_train_loss['Reconstruction_Loss']
        train_recon_loss_per_batch[batch_idx] = train_recon_loss
        train_kld_loss = joint_train_loss['KLD'] + omic1_train_loss['KLD'] + omic2_train_loss['KLD']
        train_kl_loss_per_batch[batch_idx] = train_kld_loss

        # compute and take gradient step
        train_loss.backward()
        optimizer.step()

        progress_bar.update()

    progress_bar.close()
    if epoch % args['log_interval'] == 0:
        tf_logger.add_scalar("train loss", train_loss_per_batch.mean(), epoch)
        tf_logger.add_scalar("train reconstruction loss", train_recon_loss_per_batch.mean(), epoch)
        tf_logger.add_scalar("train KL loss", train_kl_loss_per_batch.mean(), epoch)

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_per_batch.mean()))
        print('====> Epoch: {}\tReconstruction Loss: {:.4f}'.format(epoch, train_recon_loss_per_batch.mean()))
        print('====> Epoch: {}\tKLD Loss: {:.4f}'.format(epoch, train_kl_loss_per_batch.mean()))


def test(args, model, val_loader, optimizer, epoch, tf_logger):
    model.training = False
    model.eval()
    validation_loss = 0
    validation_recon_loss = 0
    validation_kl_loss = 0

    for batch_idx, (omic1, omic2) in enumerate(val_loader):
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


def run(args) -> None:
    args['cuda'] = args['cuda'] and torch.cuda.is_available()

    # random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], '{}'.format('MoE' if args['mixture'] else 'PoE'))
    os.makedirs(save_dir)

    # Define tensorboard logger
    tf_logger = SummaryWriter(save_dir)

    # Fetch Datasets
    tcga_data = datasets.TCGAData(args, save_dir=save_dir)
    train_dataset = tcga_data.get_data_partition("train")
    val_dataset = tcga_data.get_data_partition("val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  # (1 batch)

    total_batches = len(train_loader)
    total_val_batches = len(val_loader)  # Should be 1

    # Setup and log model
    model = MVAE(use_mixture=args['mixture'], latent_dim=args['latent_dim'], use_cuda=args['cuda'])

    model = model.double()

    # Log Data shape, input arguments and model
    model_file = open("{}/MVAE {} Model.txt".format(save_dir, 'MoE' if args['mixture'] else 'PoE'), "a")
    model_file.write("Running at {}\n".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    model_file.write("Input shape : {}, 5000\n".format(len(train_loader.dataset)))
    model_file.write("Input args : {}\n".format(args))
    model_file.write("PoE Model : {}".format(model))
    model_file.close()

    # Preparation for training
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # Setup early stopping, terminates training when validation loss does not improve for early_stopping_patience epochs
    early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

    if args['cuda']:
        model.cuda()

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, train_loader, optimizer, epoch, tf_logger)
        validation_loss = test(args, model, val_loader, optimizer, epoch, tf_logger)

        # Save the last model
        if epoch == args['epochs'] or epoch % args['log_save_interval'] == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_loss': validation_loss,
                'latent_dim': args['latent_dim'],
                'epochs': args['epochs'],
                'lr': args['lr'],
                'batch_size': args['batch_size'],
                'use_mixture': args['mixture'],
                'optimizer': optimizer.state_dict(),
            }, epoch, save_dir)

        early_stopping(validation_loss)

        # Stop training when not improving
        if early_stopping.early_stop:
            logger.info('Early stopping training since loss did not improve for {} epochs.'
                        .format(args['early_stopping_patience']))
            args['epochs'] = epoch  # Update nr of epochs for plots
            break

    # Extract Phase #
    logger.success("Finished training MVAE model. Now calculating task results.")

    # Imputation
    if args['task'] == 1:
        logger.info("Task 1 Imputation: Extracting Z using test set")

        impute_dataset = tcga_data.get_data_partition("test")

        # 1 batch (whole test set)
        impute_loader = torch.utils.data.DataLoader(impute_dataset, batch_size=len(impute_dataset), shuffle=False)

        omic1_from_joint, omic2_from_joint, \
        omic1_from_omic1, omic2_from_omic1, \
        omic1_from_omic2, omic2_from_omic2 = impute(model, impute_loader)

        # Reconstruction losses
        omic1_joint_reconstruction_loss = mean_squared_error(omic1_from_joint, impute_dataset.omic1_data)
        omic1_reconstruction_loss = mean_squared_error(omic1_from_omic1, impute_dataset.omic1_data)

        omic2_joint_reconstruction_loss = mean_squared_error(omic2_from_joint, impute_dataset.omic2_data)
        omic2_reconstruction_loss = mean_squared_error(omic2_from_omic2, impute_dataset.omic2_data)
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data1'], "both omics", omic1_joint_reconstruction_loss))
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data1'], args['data1'], omic1_reconstruction_loss))
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data2'], "both omics", omic2_joint_reconstruction_loss))
        logger.info("Reconstruction loss for {} from {} : {}".
                    format(args['data2'], args['data2'], omic2_reconstruction_loss))

        # Imputation losses
        omic1_imputation_loss = mean_squared_error(omic1_from_omic2, impute_dataset.omic1_data)
        omic2_imputation_loss = mean_squared_error(omic2_from_omic1, impute_dataset.omic2_data)
        logger.info("Imputation loss for {} from {} : {}".
                    format(args['data1'], args['data2'], omic1_imputation_loss))
        logger.info("Imputation loss for {} from {} : {}".
                    format(args['data2'], args['data1'], omic2_imputation_loss))

        # Get embeddings for UMAP
        for omic1, omic2 in impute_loader: # Runs once since there is 1 batch
            z = model.extract(omic1, omic2)
            z = z.detach().numpy()
            np.save("{}/task1_z.npy".format(save_dir), z)

            label_types, labels, test_ind = tcga_data.get_labels_partition("test")

            labels = labels[test_ind].astype(int)
            sample_labels = label_types[[labels]]

            plot = UMAPPlotter(z, sample_labels, "{}: Task {} | {} & {} \n"
                                                 "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                               .format('MoE' if args['mixture'] else 'PoE', args['task'], args['data1'], args['data2'],
                                       args['epochs'], args['latent_dim'], args['lr'], args['batch_size']),
                               save_dir + "/{} UMAP.png".format('MoE' if args['mixture'] else 'PoE', 'MoE' if args['mixture'] else 'PoE'))

            plot.plot()

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
