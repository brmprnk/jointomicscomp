import os
from datetime import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Independent, Laplace
from tensorboardX import SummaryWriter
import pickle

from src.MoE.model import MixtureOfExperts
import src.PoE.datasets as datasets
from src.PoE.evaluate import impute
from src.CGAE.model import train, MultiOmicsDataset, evaluateUsingBatches, evaluatePerDatapoint
import src.util.logger as logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter
from src.util.evaluate import evaluate_imputation, save_factorizations_to_csv
from src.baseline.baseline import classification, classificationMLP

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

    n_modalities = args['nomics']

    # Load in data
    omics = [np.load(args['data_path%d' % (i+1)]) for i in range(n_modalities)]

    labels = np.load(args['labels'])
    labeltypes = np.load(args['labelnames'], allow_pickle=True)

    # Use predefined split
    train_ind = np.load(args['train_ind'])
    val_ind = np.load(args['val_ind'])
    test_ind = np.load(args['test_ind'])

    omics_train = [omic[train_ind] for omic in omics]
    omics_val = [omic[val_ind] for omic in omics]
    omics_test = [omic[test_ind] for omic in omics]

    ytrain = labels[train_ind]
    yvalid = labels[val_ind]
    ytest = labels[test_ind]

    # Number of features
    input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

    likelihoods = [args['likelihood%d' % (i+1)] for i in range(n_modalities)]

    llikScales = [args['llikescale%d' % (i+1)] for i in range(n_modalities)]

    dataTrain = [torch.tensor(omic, device=device) for omic in omics_train]
    dataValidation = [torch.tensor(omic, device=device) for omic in omics_val]
    dataTest = [torch.tensor(omic, device=device) for omic in omics_test]

    datasetTrain = MultiOmicsDataset(dataTrain)
    datasetValidation = MultiOmicsDataset(dataValidation)


    train_loader = torch.utils.data.DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=False)

    if args['train_loader_eval_batch_size'] > 0:
        trnEvalBatchSize = args['train_loader_eval_batch_size']
        valBatchSize = trnEvalBatchSize
    else:
        trnEvalBatchSize = len(datasetTrain)
        valBatchSize = dataValidation1.shape[0]

    train_loader_eval = torch.utils.data.DataLoader(datasetTrain, batch_size=trnEvalBatchSize, shuffle=False)
    val_loader = torch.utils.data.DataLoader(datasetValidation, batch_size=valBatchSize, shuffle=False, num_workers=0, drop_last=False)

    encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
    decoder_layers = encoder_layers[::-1][1:]


    if 'categorical' in likelihoods:
        categories = [args['n_categories%d' % (i + 1)] for i in range(n_modalities)]
    else:
        categories = None


    model = MixtureOfExperts(input_dims, encoder_layers, decoder_layers,
     likelihoods, args['use_batch_norm'],
     args['dropout_probability'], args['optimizer'], args['lr'], args['lr'],
     args['enc_distribution'], args['beta_start_value'], args['K'], llikScales, categories)

    model.double()
    if device == torch.device('cuda'):
        model.cuda()
    else:
        args['cuda'] = False

    if 'pre_trained' in args and args['pre_trained'] != '':
        checkpoint = torch.load(args['pre_trained'])

        for i in range(n_modalities):
            #print(i)
            model.encoders[i].load_state_dict(checkpoint['state_dict_enc'][i])
            model.decoders[i].load_state_dict(checkpoint['state_dict_dec'][i])

        logger.success("Loaded trained MixtureOfExperts model.")

    else:

        # Log Data shape, input arguments and model
        model_file = open("{}/MoE_Model.txt".format(save_dir), "a")
        model_file.write("Running at {}\n".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
        model_file.write("Input shape 1 : {}, {}\n".format(len(train_loader.dataset), args['num_features1']))
        model_file.write("Input shape 2 : {}, {}\n".format(len(train_loader.dataset), args['num_features2']))
        model_file.write("Input args : {}\n".format(args))
        model_file.write("MoE Model : {}".format(model))
        model_file.close()

        ckpt_dir = save_dir + '/checkpoint'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        logs_dir = save_dir + '/logs'

        # Setup early stopping, terminates training when validation loss does not improve for early_stopping_patience epochs
        early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

        bestLoss, bestEpoch = train(device=device, net=model, num_epochs=args['epochs'], train_loader=train_loader,
              train_loader_eval=train_loader_eval, valid_loader=val_loader,
              ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping, save_step=args['log_save_interval'], multimodal=True)

        logger.info("Using model from epoch %d" % bestEpoch)

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
    # logger.success("Finished training MVAE model. Now calculating task results.")


    if args['task'] == 0:
        lossDict = {'epoch': bestEpoch, 'val_loss': bestLoss}
        with open(save_dir + '/finalValidationLoss.pkl', 'wb') as f:
            pickle.dump(lossDict, f)



    # Imputation
    if args['task'] > 0:
        dataTrain = [torch.tensor(omic1, device=device) for omic1 in omics_train]
        dataValidation = [torch.tensor(omic1, device=device) for omic1 in omics_val]
        dataTest = [torch.tensor(omic1, device=device) for omic1 in omics_test]

        datasetTrain = MultiOmicsDataset(dataTrain)
        datasetValidation = MultiOmicsDataset(dataValidation)
        datasetTest = MultiOmicsDataset(dataTest)

        train_loader = torch.utils.data.DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=False, num_workers=0,
                                  drop_last=False)

        valid_loader = torch.utils.data.DataLoader(datasetValidation, batch_size=args['batch_size'], shuffle=False, num_workers=0,
                                  drop_last=False)

        test_loader = torch.utils.data.DataLoader(datasetTest, batch_size=args['batch_size'], shuffle=False, num_workers=0,
                                  drop_last=False)

        test_loader_individual = torch.utils.data.DataLoader(datasetTest, batch_size=1, shuffle=False, num_workers=0,
                                  drop_last=False)


        ztrain = [np.zeros((dataTrain[i].shape[0], model.z_dim)) for i in range(n_modalities)]
        zvalidation = [np.zeros((dataValidation[i].shape[0], model.z_dim)) for i in range(n_modalities)]
        ztest = [np.zeros((dataTest[i].shape[0], model.z_dim)) for i in range(n_modalities)]

        model.eval()

        ind = 0
        b = args['batch_size']
        for data in train_loader:
            batch = (data[0][0].double(), data[1][0].double())

            z_tmp, _ = model.embedAndReconstruct(batch)

            for ii in range(n_modalities):
                ztrain[ii][ind:ind+b] = z_tmp[ii].cpu().detach().numpy()

            ind += b

        ind = 0
        b = val_loader.batch_size
        for data in val_loader:
            batch = (data[0][0].double(), data[1][0].double())
            z_tmp, _ = model.embedAndReconstruct(batch)

            for ii in range(n_modalities):
                zvalidation[ii][ind:ind+b] = z_tmp[ii].cpu().detach().numpy()

            ind += b


        ind = 0
        b = test_loader.batch_size
        for data in test_loader:
            batch = (data[0][0].double(), data[1][0].double())
            z_tmp, _ = model.embedAndReconstruct(batch)

            for ii in range(n_modalities):
                ztest[ii][ind:ind+b] = z_tmp[ii].cpu().detach().numpy()

            ind += b

        zrand = Independent(Laplace(torch.zeros(model.z_dim).to(torch.device('cuda:0')), torch.ones(model.z_dim).to(torch.device('cuda:0'))), 1).sample([2000])
        zrand = zrand.double()

        Xsample = [dec(zrand).mean.cpu().detach() for dec in model.decoders]


        logger.info('Evaluating...')
        from src.util.evaluate import evaluate_imputation, evaluate_classification, evaluate_generation

        if args['nomics'] == 2:
            logger.info('Generation coherence')
            acc = evaluate_generation(Xsample[0], Xsample[1], args['data1'], args['data2'])
            logger.info('Concordance: %.4f: ' % acc)
            logger.info('\n\n')


        logger.info('Validation set:')
        metricsValidation = evaluateUsingBatches(model, device, val_loader, True)
        for m in metricsValidation:
            logger.info('%s\t%.4f' % (m, metricsValidation[m]))

        logger.info('Test set:')
        metricsTest = evaluateUsingBatches(model, device, test_loader, True)
        for m in metricsTest:
            logger.info('%s\t%.4f' % (m, metricsTest[m]))

        metricsTestIndividual = evaluatePerDatapoint(model, device, test_loader_individual, True)
        # for m in metricsTest:
        #     logger.info('%s\t%.4f' % (m, torch.mean(metricsTestIndividual[m])))

        logger.info('Saving individual performances...')

        with open(save_dir + '/test_performance_per_datapoint.pkl', 'wb') as f:
            pickle.dump(metricsTestIndividual, f)




        logger.info('Saving embeddings...')

        with open(save_dir + '/embeddings.pkl', 'wb') as f:
            embDict = {'ztrain': ztrain, 'zvalidation': zvalidation, 'ztest': ztest}
            pickle.dump(embDict, f)


    if args['task'] > 1:
        assert args['nomics'] == 2
        classLabels = np.load(args['labels'])
        labelNames = np.load(args['labelnames'], allow_pickle=True)

        ytrain = classLabels[train_ind]
        yvalid = classLabels[val_ind]
        ytest = classLabels[test_ind]


        logger.info('Test performance, classification task, linear classifier, modality 1')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(ztrain[0], ytrain, zvalidation[0], yvalid, ztest[0], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
        performance1 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


        pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

        logger.info('Test performance, classification task, linear classifier, modality 2')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(ztrain[1], ytrain, zvalidation[1], yvalid, ztest[1], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])
        performance2 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

        pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

        logger.info('Test performance, classification task, linear classifier, both modalities')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(np.hstack((ztrain[0], ztrain[1])), ytrain, np.hstack((zvalidation[0], zvalidation[1])), yvalid, np.hstack((ztest[0], ztest[1])), ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])
        performance12 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

        pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}

        if 'level' in args:
            level = args['level']
            assert level == 'l3'
        else:
            level = 'l2'

        # -----------------------------------------------------------------
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(ztrain[0], ytrain, zvalidation[0], yvalid, ztest[0], ytest, 'type-classifier/eval/' + level + '/moe_' + args['data1'] + '/')
        performance1 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('Test performance, classification task, non-linear classifier, modality 1')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


        mlp_pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(ztrain[1], ytrain, zvalidation[1], yvalid, ztest[1], ytest, 'type-classifier/eval/' + level + '/moe_' + args['data2'] + '/')
        performance2 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('Test performance, classification task, non-linear classifier, modality 2')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

        mlp_pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(np.hstack((ztrain[0], ztrain[1])), ytrain, np.hstack((zvalidation[0], zvalidation[1])), yvalid, np.hstack((ztest[0], ztest[1])), ytest, 'type-classifier/eval/' + level + '/moe_' + args['data1'] + '_' + args['data2'] + '/')
        performance12 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('Test performance, classification task, non-linear classifier, both modalities')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

        mlp_pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}


        logger.info("Saving results")
        with open(save_dir + "/CGAE_task2_results.pkl", 'wb') as f:
            pickle.dump({'omic1': pr1, 'omic2': pr2, 'omic1+2': pr12, 'omic1-mlp': mlp_pr1, 'omic2-mlp': mlp_pr2, 'omic1+2-mlp': mlp_pr12}, f)
