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
    with open('deleteme.txt', 'w') as fw:
        fw.write('Omic1: %d x %d\n' % (omic1.shape[0], omic1.shape[1]))
        fw.write('Omic1 rec: %d x %d\n' % (recon_omic1.shape[0], recon_omic1.shape[1]))


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
    torch.save(state, os.path.join(save_dir, 'model_epoch{}.pth.tar'.format(epoch)))


def train(args, model, train_loader, optimizer, epoch, tf_logger):
    model.training = True
    model = model.double()
    model.train()

    #progress_bar = tqdm(total=len(train_loader))
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

        #progress_bar.update()

    #progress_bar.close()
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
    validation_joint_loss = 0
    validation_joint_recon_loss = 0
    validation_joint_kl_loss = 0

    validation_omic1_loss = 0
    validation_omic1_recon_loss = 0
    validation_omic1_kl_loss = 0

    validation_omic2_loss = 0
    validation_omic2_recon_loss = 0
    validation_omic2_kl_loss = 0


    for batch_idx, (omic1, omic2) in enumerate(val_loader):

        if args['cuda']:
            omic1 = omic1.cuda()
            omic2 = omic2.cuda()

        with torch.no_grad():
            (joint_recon_omic1, joint_recon_omic2, joint_mu, joint_logvar) = model.forward(omic1, omic2)

            # compute reconstructions using each of the individual modalities
            (omic1_recon_omic1, omic1_recon_omic2, omic1_mu, omic1_logvar) = model.forward(omic1=omic1)

            (omic2_recon_omic1, omic2_recon_omic2, omic2_mu, omic2_logvar) = model.forward(omic2=omic2)



        kld_weight = len(omic1) / len(val_loader.dataset)  # Account for the minibatch samples from the dataset

        # Compute joint loss
        joint_test_loss = loss_function(joint_recon_omic1, omic1,
                                        joint_recon_omic2, omic2,
                                        joint_mu, joint_logvar, kld_weight)

        omic1_test_loss = loss_function(omic1_recon_omic1, omic1,
                                        omic1_recon_omic2, omic2,
                                        omic1_mu, omic1_logvar, kld_weight)

        omic2_test_loss = loss_function(omic2_recon_omic1, omic1,
                                        omic2_recon_omic2, omic2,
                                        omic2_mu, omic2_logvar, kld_weight)


        validation_joint_loss += joint_test_loss['loss']
        validation_joint_recon_loss += joint_test_loss['Reconstruction_Loss']
        validation_joint_kl_loss += joint_test_loss['KLD']

        validation_omic1_loss += omic1_test_loss['loss']
        validation_omic1_recon_loss += omic1_test_loss['Reconstruction_Loss']
        validation_omic1_kl_loss += omic1_test_loss['KLD']

        validation_omic2_loss += omic2_test_loss['loss']
        validation_omic2_recon_loss += omic2_test_loss['Reconstruction_Loss']
        validation_omic2_kl_loss += omic2_test_loss['KLD']

    validation_joint_loss /= len(val_loader)
    validation_joint_recon_loss /= len(val_loader)
    validation_joint_kl_loss /= len(val_loader)

    validation_omic1_loss /= len(val_loader)
    validation_omic1_recon_loss /= len(val_loader)
    validation_omic1_kl_loss /= len(val_loader)

    validation_omic2_loss /= len(val_loader)
    validation_omic2_recon_loss /= len(val_loader)
    validation_omic2_kl_loss /= len(val_loader)

    if epoch > 0:
        tf_logger.add_scalar("loss/joint/validation", validation_joint_loss, epoch)
        tf_logger.add_scalar("mse/joint/validation", validation_joint_recon_loss, epoch)
        tf_logger.add_scalar("KL/joint/validation", -1 * validation_joint_kl_loss, epoch)

        tf_logger.add_scalar("loss/1/validation", validation_omic1_loss, epoch)
        tf_logger.add_scalar("mse/1/validation", validation_omic1_recon_loss, epoch)
        tf_logger.add_scalar("KL/1/validation", -1 * validation_omic1_kl_loss, epoch)

        tf_logger.add_scalar("loss/2/validation", validation_omic2_loss, epoch)
        tf_logger.add_scalar("mse/2/validation", validation_omic2_recon_loss, epoch)
        tf_logger.add_scalar("KL/2/validation", -1 * validation_omic2_kl_loss, epoch)

    validation_loss = validation_joint_loss + validation_omic1_loss + validation_omic2_loss

    print('====> Epoch: {}\tValidation Loss: {:.4f}'.format(epoch, validation_loss))


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

    save_dir = os.path.join(args['save_dir'], 'PoE/')
    os.makedirs(save_dir)

    # Define tensorboard logger
    tf_logger = SummaryWriter(save_dir + '/logs')
    ckpt_dir = save_dir + '/checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


    # Fetch Datasets
    tcga_data = datasets.TCGAData(args, save_dir=save_dir)
    train_dataset = tcga_data.get_data_partition("train")
    val_dataset = tcga_data.get_data_partition("val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  # (1 batch)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)  #

    # Setup and log model
    device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')

    model = PoE(args)
    if device == torch.device('cuda'):
        model.cuda()
    else:
        args['cuda'] = False

    model.double()

    if 'pre_trained' in args and args['pre_trained'] != '':
        checkpoint = torch.load(args['pre_trained'])

        model.load_state_dict(checkpoint['state_dict'])
        logger.success("Loaded trained ProductOfExperts model.")

    else:


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

        checkpoint_loss = [test(args, model, val_loader, optimizer, 0, tf_logger)]

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
                }, epoch, ckpt_dir)
                checkpoint_loss.append(validation_loss)

            early_stopping(validation_loss)

            # Stop training when not improving
            if early_stopping.early_stop:
                logger.info('Early stopping training since loss did not improve for {} epochs.'
                            .format(args['early_stopping_patience']))
                args['epochs'] = epoch  # Update nr of epochs for plots
                break


        logger.success("Finished training PoE model.")

        # find best checkpoint based on the validation loss
        bestEpoch = args['log_save_interval'] * np.argmin(checkpoint_loss)

        logger.info("Using model from epoch %d" % bestEpoch)
        modelCheckpoint = ckpt_dir + '/model_epoch%d.pth.tar' % (bestEpoch)
        assert os.path.exists(modelCheckpoint)


    # Extract Phase #

    if args['task'] == 0:
        lossDict = {'epoch': bestEpoch, 'val_loss': np.min(checkpoint_loss)}
        with open(save_dir + '/finalValidationLoss.pkl', 'wb') as f:
            pickle.dump(lossDict, f)




    # Imputation
    if args['task'] > 0:


        # Fetch Datasets
        tcga_data = datasets.TCGAData(args, save_dir=save_dir)
        train_dataset = tcga_data.get_data_partition("train")
        val_dataset = tcga_data.get_data_partition("val")
        test_dataset = tcga_data.get_data_partition("test")

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False)  #
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False)  #

        ztrain = np.zeros((len(train_loader.dataset), model.latent_dim))

        zvalidation = np.zeros((len(val_loader.dataset), model.latent_dim))

        ztest = np.zeros((len(test_loader.dataset), model.latent_dim))

        x1_hat_train = np.zeros((len(train_loader.dataset), args['num_features1']))
        x2_hat_train = np.zeros((len(train_loader.dataset), args['num_features2']))

        x1_hat_validation = np.zeros((len(val_loader.dataset), args['num_features1']))
        x2_hat_validation = np.zeros((len(val_loader.dataset), args['num_features2']))

        x1_hat_test = np.zeros((len(test_loader.dataset), args['num_features1']))
        x2_hat_test = np.zeros((len(test_loader.dataset), args['num_features2']))

        x1_cross_hat_train = np.zeros((len(train_loader.dataset), args['num_features1']))
        x2_cross_hat_train = np.zeros((len(train_loader.dataset), args['num_features2']))

        x1_cross_hat_validation = np.zeros((len(val_loader.dataset), args['num_features1']))
        x2_cross_hat_validation = np.zeros((len(val_loader.dataset), args['num_features2']))

        x1_cross_hat_test = np.zeros((len(test_loader.dataset), args['num_features1']))
        x2_cross_hat_test = np.zeros((len(test_loader.dataset), args['num_features2']))

        model.eval()

        ind = 0
        b = args['batch_size']
        for (b1, b2) in train_loader:
            b1 = b1.to(device)
            b2 = b2.to(device)

            z_tmp, x1_hat_tmp, x2_hat_tmp, x1_cross_hat_tmp, x2_cross_hat_tmp = model.embedAndReconstruct(b1, b2)

            ztrain[ind:ind+b] = z_tmp.cpu().detach().numpy()

            x1_hat_train[ind:ind+b] = x1_hat_tmp.cpu().detach().numpy()
            x2_hat_train[ind:ind+b] = x2_hat_tmp.cpu().detach().numpy()

            x1_cross_hat_train[ind:ind+b] = x1_cross_hat_tmp.cpu().detach().numpy()
            x2_cross_hat_train[ind:ind+b] = x2_cross_hat_tmp.cpu().detach().numpy()

            ind += b

        ind = 0
        for (b1, b2) in val_loader:
            b1 = b1.to(device)
            b2 = b2.to(device)

            z_tmp, x1_hat_tmp, x2_hat_tmp, x1_cross_hat_tmp, x2_cross_hat_tmp = model.embedAndReconstruct(b1, b2)

            zvalidation[ind:ind+b] = z_tmp.cpu().detach().numpy()

            x1_hat_validation[ind:ind+b] = x1_hat_tmp.cpu().detach().numpy()
            x2_hat_validation[ind:ind+b] = x2_hat_tmp.cpu().detach().numpy()

            x1_cross_hat_validation[ind:ind+b] = x1_cross_hat_tmp.cpu().detach().numpy()
            x2_cross_hat_validation[ind:ind+b] = x2_cross_hat_tmp.cpu().detach().numpy()

            ind += b


        ind = 0
        for (b1, b2) in test_loader:
            b1 = b1.to(device)
            b2 = b2.to(device)

            z_tmp, x1_hat_tmp, x2_hat_tmp, x1_cross_hat_tmp, x2_cross_hat_tmp = model.embedAndReconstruct(b1, b2)

            ztest[ind:ind+b] = z_tmp.cpu().detach().numpy()

            x1_hat_test[ind:ind+b] = x1_hat_tmp.cpu().detach().numpy()
            x2_hat_test[ind:ind+b] = x2_hat_tmp.cpu().detach().numpy()

            x1_cross_hat_test[ind:ind+b] = x1_cross_hat_tmp.cpu().detach().numpy()
            x2_cross_hat_test[ind:ind+b] = x2_cross_hat_tmp.cpu().detach().numpy()

            ind += b

        # draw random samples from the prior and reconstruct them
        zrand = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(model.latent_dim), torch.ones(model.latent_dim)), 1).sample([2000]).to(device)
        zrand = zrand.double()


        X1sample = model.omic1_decoder(zrand).cpu().detach()
        X2sample = model.omic2_decoder(zrand).cpu().detach()


        data1 = np.load(args['data_path1'])
        data2 = np.load(args['data_path2'])
        labels = np.load(args['labels'])

        trainInd = np.load(args['train_ind'])
        validInd = np.load(args['val_ind'])
        testInd = np.load(args['test_ind'])

        dataTrain1 = data1[trainInd]
        dataValidation1 = data1[validInd]
        dataTest1 = data1[testInd]

        dataTrain2 = data2[trainInd]
        dataValidation2 = data2[validInd]
        dataTest2 = data2[testInd]

        from src.util.evaluate import evaluate_imputation, evaluate_classification, evaluate_generation
        logger.info('Evaluating...')

        logger.info('Training performance, reconstruction error, modality 1')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTrain1, x1_hat_train, data1.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Training performance, reconstruction error, modality 2')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTrain2, x2_hat_train, data2.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Training performance, imputation error, modality 1')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTrain1, x1_cross_hat_train, data1.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Training performance, imputation error, modality 2')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTrain2, x2_cross_hat_train, data2.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Validation performance, reconstruction error, modality 1')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataValidation1, x1_hat_validation, data1.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Validation performance, reconstruction error, modality 2')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataValidation2, x2_hat_validation, data2.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Validation performance, imputation error, modality 1')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataValidation1, x1_cross_hat_validation, data1.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Validation performance, imputation error, modality 2')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataValidation2, x2_cross_hat_validation, data2.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Test performance, reconstruction error, modality 1')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTest1, x1_hat_test, data1.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Test performance, reconstruction error, modality 2')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTest2, x2_hat_test, data2.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Test performance, imputation error, modality 1')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTest1, x1_cross_hat_test, data1.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Test performance, imputation error, modality 2')
        mse_train, spearman_train, r2_train = evaluate_imputation(dataTest2, x2_cross_hat_test, data2.shape[1])
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse_train, spearman_train, r2_train))

        logger.info('Generation coherence')
        acc = evaluate_generation(X1sample, X2sample, args['data1'], args['data2'])
        logger.info('Concordance: %.4f: ' % acc)


        logger.info('Saving embeddings...')

        with open(save_dir + '/embeddings.pkl', 'wb') as f:
            embDict = {'ztrain': ztrain, 'zvalidation': zvalidation, 'ztest': ztest}
            pickle.dump(embDict, f)

    if args['task'] > 1:

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


        logger.info('Test performance, classification task, both modalities')
        predictions, acc, pr, rc, f1, mcc, confMat = classification(ztrain, ytrain, zvalidation, yvalid, ztest, ytest, np.array([1e-5, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])
        performance = [acc, pr, rc, f1, mcc, confMat]
        pr = {'acc': performance[0], 'pr': performance[1], 'rc': performance[2], 'f1': performance[3], 'mcc': performance[4], 'confmat': performance[5]}
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance[0], np.mean(performance[1]), np.mean(performance[2]), np.mean(performance[3]), performance[4]))


        logger.info("Saving results")
        with open(save_dir + "/PoE_task2_results.pkl", 'wb') as f:
            pickle.dump({'joint': pr}, f)
