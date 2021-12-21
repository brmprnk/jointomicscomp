import os
from datetime import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pickle

from src.PoE.model import PoE
import src.PoE.datasets as datasets
from src.PoE.evaluate import impute
import src.util.logger as logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter
from src.util.evaluate import evaluate_imputation, save_factorizations_to_csv
from src.baseline.baseline import classification

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
    model = model.double()
    model.train()

    progress_bar = tqdm(total=len(train_loader))
    train_loss_per_batch = np.zeros(len(train_loader))
    train_recon_loss_per_batch = np.zeros(len(train_loader))
    train_kl_loss_per_batch = np.zeros(len(train_loader))

    # Incorporate MMVAE training function
    for batch_idx, (omic1, omic2) in enumerate(train_loader):

        if args['cuda']:
            omic1 = omic1.cuda()
            omic2 = omic2.cuda()

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
    if epoch % args['log_save_interval'] == 0:
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

    if epoch % args['log_save_interval'] == 0:

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

    trained_model = PoE(args)
    trained_model.load_state_dict(checkpoint['state_dict'])
    return trained_model, checkpoint


def run(args) -> None:
    # # random seed
    # # https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.benchmark = True
    # torch.manual_seed(args['random_seed'])
    # np.random.seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], 'PoE')
    os.makedirs(save_dir)

    # Define tensorboard logger
    tf_logger = SummaryWriter(save_dir)

    # Fetch Datasets
    tcga_data = datasets.TCGAData(args, save_dir=save_dir)
    train_dataset = tcga_data.get_data_partition("train")
    val_dataset = tcga_data.get_data_partition("val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  # (1 batch)

    if args['pre_trained'] != "":
        logger.info("Using Pre-Trained PoE found at {}".format(args['pre_trained']))

        save_dir = os.path.dirname(args['pre_trained'])

        print("-----   Loading Trained Model   -----")
        model, checkpoint = load_checkpoint(args, args['cuda'])
        model.eval()

    else:
        # Setup and log model
        device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')

        model = PoE(args)
        if device == torch.device('cuda'):
            model.cuda()
        else:
            args['cuda'] = False

        # Log Data shape, input arguments and model
        model_file = open("{}/PoE {} Model.txt".format(save_dir, 'PoE'), "a")
        model_file.write("Running at {}\n".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
        model_file.write("Input shape 1 : {}, {}\n".format(len(train_loader.dataset), args['num_features1']))
        model_file.write("Input shape 2 : {}, {}\n".format(len(train_loader.dataset), args['num_features2']))
        model_file.write("Input args : {}\n".format(args))
        model_file.write("PoE Model : {}".format(model))
        model_file.close()

        # Preparation for training
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])

        # Setup early stopping, terminates training when validation loss does not improve for early_stopping_patience epochs
        early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

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
    logger.success("Finished training PoE model. Now calculating task results.")

    # Imputation
    if args['task'] == 1:
        # Correct me if I'm wrong, but we can just get the x1_cross_hat and x2_cross_hat
        # from the model.forward

        logger.info("Task 1 Imputation: Extracting Z using test set")

        impute_dataset = tcga_data.get_data_partition("test")

        # Deallocate tensors still on GPU from training
        if args['cuda']:
            torch.cuda.empty_cache()


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
        with open(save_dir + "/PoE_results_pickle", 'wb') as f:
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
            sample_labels = label_types[tuple([labels])]

            plot = UMAPPlotter(z, sample_labels, "PoE: Task {} | {} & {} \n"
                                                 "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                               .format(args['task'], args['data1'], args['data2'],
                                       29, args['latent_dim'], args['lr'], args['batch_size']),
                               save_dir + "/PoE_UMAP.png")

            plot.plot()

    if args['task'] == 2:
        print(model)
        logger.success("Extract z1 and z2 for classification")

        train_dataset = tcga_data.get_data_partition("train")
        y, _, trnInd = tcga_data.get_labels_partition("train")
        ytrain = y[trnInd]
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

        valid_dataset = tcga_data.get_data_partition("val")
        y, _, valInd = tcga_data.get_labels_partition("val")
        yvalid = y[valInd]
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

        test_dataset = tcga_data.get_data_partition("test")
        y, _, tstInd = tcga_data.get_labels_partition("test")
        ytest = y[tstInd]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


        for GE, ME in train_loader:
            if args['cuda']:
                GE = GE.cuda()
                ME = ME.cuda()

            ztrain = model.extract(GE, ME)

        for GE, ME in valid_loader:
            if args['cuda']:
                GE = GE.cuda()
                ME = ME.cuda()

            zvalid = model.extract(GE, ME)

        for GE, ME in test_loader:
            if args['cuda']:
                GE = GE.cuda()
                ME = ME.cuda()

            ztest = model.extract(GE, ME)

        if args['cuda']:
            ztrain = ztrain.detach().cpu().numpy()
            zvalid = zvalid.detach().cpu().numpy()
            ztest = ztest.detach().cpu().numpy()
        else:
            ztrain = ztrain.detach().numpy()
            zvalid = zvalid.detach().numpy()
            ztest = ztest.detach().numpy()


        predictions, performance = classification(ztrain, ytrain, zvalid, yvalid, ztest, ytest, np.array([1e-5, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])
        pr = {'acc': performance[0], 'pr': performance[1], 'rc': performance[2], 'f1': performance[3], 'mcc': performance[4], 'confmat': performance[5], 'pred': predictions}


        logger.info("Saving results")
        with open(save_dir + "/PoE_task2_results.pkl", 'wb') as f:
            pickle.dump({'joint': pr}, f)
