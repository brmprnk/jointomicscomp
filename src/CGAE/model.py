import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import src.nets
from sklearn.metrics import mean_squared_error
from src.util import logger
from src.util.evaluate import evaluate_imputation, save_factorizations_to_csv
from src.MoE.model import MixtureOfExperts

class MultiOmicsDataset():
    def __init__(self, data):
        self.data = [TensorDataset(d) for d in data]
        # load the 2nd data view
        self.length = data[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return [d1.__getitem__(index) for d1 in self.data]

def multiomic_collate(batch):
    d1 = [x[0] for x in batch]
    d2 = [x[1] for x in batch]
    print(len(d1))
    print(len(d1[0]))

    return torch.from_numpy(np.array(d1)), torch.from_numpy(np.array(d2))

def evaluateUsingBatches(net, device, dataloader, multimodal=False):
    for i, trd in enumerate(dataloader):
        if i == 0:
            if not multimodal:
                x = trd[0].to(device).double()
                metrics = net.evaluate(x)
                for kk in metrics:
                    metrics[kk] *= x.shape[0]
            else:
                x1 = trd[0][0].to(device).double()
                x2 = trd[1][0].to(device).double()
                # evaluate method averages across samples, multiply with #samples to get total
                metrics = net.evaluate([x1, x2])
                for kk in metrics:
                    metrics[kk] *= x1.shape[0]

        else:
            # add intermediate metrics to total
            if not multimodal:
                x = trd[0].to(device).double()
                tmpmetrics = net.evaluate(x)
                shape = x.shape[0]
            else:
                x1 = trd[0][0].to(device).double()
                x2 = trd[1][0].to(device).double()
                tmpmetrics = net.evaluate([x1, x2])
                shape = x1.shape[0]

            for kk in metrics:
                metrics[kk] += tmpmetrics[kk] * shape

    for kk in metrics:
        # now divide by total number of points to get the average across all samples
        metrics[kk] /= len(dataloader.dataset)


    return metrics

def evaluatePerDatapoint(net, device, dataloader, multimodal=False):
    assert len(dataloader) == len(dataloader.dataset), 'Use batch size 1 for evaluatePerDatapoint'
    metrics = dict()
    for i, trd in enumerate(dataloader):
        if i == 0:
            if not multimodal:
                raise NotImplementedError
                # x = trd[0].to(device).double()
                # metrics = net.evaluate(x)
                # for kk in metrics:
                #     metrics[kk] *= x.shape[0]
            else:
                x1 = trd[0][0].to(device).double().reshape(1,-1)
                x2 = trd[1][0].to(device).double().reshape(1,-1)
                # evaluate method averages across samples, multiply with #samples to get total
                tmpmetrics = net.evaluate([x1, x2])


                for kk in tmpmetrics:
                    metrics[kk] = torch.zeros(len(dataloader))

        else:

            if not multimodal:
                raise NotImplementedError
            else:
                x1 = trd[0][0].to(device).double().reshape(1,-1)
                x2 = trd[1][0].to(device).double().reshape(1,-1)
                tmpmetrics = net.evaluate([x1, x2])
                shape = x1.shape[0]

        for kk in metrics:
            metrics[kk][i] = tmpmetrics[kk]


    return metrics


def train(device, net, num_epochs, train_loader, train_loader_eval, valid_loader, ckpt_dir, logs_dir, early_stopping,
          save_step=10, multimodal=False):
    # Define logger
    tf_logger = SummaryWriter(logs_dir)

    # Load checkpoint model and optimizer
    start_epoch = load_checkpoint(net, filename=ckpt_dir + '/model_last.pth.tar')


    # Evaluate validation set before start training
    print("[*] Evaluating epoch %d..." % start_epoch)
    if not isinstance(net, MixtureOfExperts):
        # this takes too long on MoE, so we skip it
        metrics = evaluateUsingBatches(net, device, train_loader_eval, multimodal)

        assert 'loss' in metrics
        print("--- Training loss:\t%.4f" % metrics['loss'])

    net.eval()
    metrics = evaluateUsingBatches(net, device, valid_loader, multimodal)
    bestValLoss = metrics['loss']
    bestValEpoch  = 0

    assert 'loss' in metrics
    print("--- Validation loss:\t%.4f" % metrics['loss'])


    # Start training phase
    print("[*] Start training...")
    # Training epochs
    for epoch in range(start_epoch, num_epochs):
        net.train()

        print("[*] Epoch %d..." % (epoch + 1))
        # for param_group in optimizer.param_groups:
        #	print('--- Current learning rate: ', param_group['lr'])

        for data in train_loader:
            # Get current batch and transfer to device
            # data = data.to(device)

            with torch.set_grad_enabled(True):  # no need to specify 'requires_grad' in tensors
                # Set the parameter gradients to zero
                net.opt.zero_grad()

                if not multimodal:
                    current_loss = net.compute_loss(data[0].to(device).double())
                else:
                    current_loss = net.compute_loss([data[0][0].to(device).double(), data[1][0].to(device).double()])

                # Backward pass and optimize
                current_loss.backward()
                net.opt.step()

        # Save last model
        state = {'epoch': epoch + 1, 'state_dict_enc': [enc.state_dict() for enc in net.encoders], 'state_dict_dec': [dec.state_dict() for dec in net.decoders], 'optimizer': net.opt.state_dict()}
        torch.save(state, ckpt_dir + '/model_last.pth.tar')


        # Evaluate all training set and validation set at epoch
        print("[*] Evaluating epoch %d..." % (epoch + 1))
        net.eval()
        if not isinstance(net, MixtureOfExperts):
            # skip for MoE, it takes too long
            metricsTrain = evaluateUsingBatches(net, device, train_loader_eval, multimodal)
            print("--- Training loss:\t%.4f" % metricsTrain['loss'])


        metricsValidation = evaluateUsingBatches(net, device, valid_loader, multimodal)
        print("--- Validation loss:\t%.4f" % metricsValidation['loss'])

        if metricsValidation['loss'] < bestValLoss:
            bestValLoss = metricsValidation['loss']
            bestValEpoch = epoch + 1
            torch.save(state, ckpt_dir + '/model_best.pth.tar')



        # Save model at epoch, and record loss at that checkpoint
        if (epoch + 1) % save_step == 0:
            print("[*] Saving model epoch %d..." % (epoch + 1))
            torch.save(state, ckpt_dir + '/model_epoch%d.pth.tar' % (epoch + 1))

        early_stopping(metricsValidation['loss'])

        for m in metricsValidation:
            if not isinstance(net, MixtureOfExperts):
                tf_logger.add_scalar(m + '/train', metricsTrain[m], epoch + 1)
            tf_logger.add_scalar(m + '/validation', metricsValidation[m], epoch + 1)




        # Stop training when not improving
        if early_stopping.early_stop:
            logger.info('Early stopping training since loss did not improve for {} epochs.'
                        .format(early_stopping.patience))
            break

    print("[*] Finish training.")
    return bestValLoss, bestValEpoch


def impute(net, model_file, loader, device, save_dir, sample_names, num_features1, num_features2, multimodal=False):
    checkpoint = torch.load(model_file)
    net.load_state_dict(checkpoint['state_dict'])
    net.opt.load_state_dict(checkpoint['optimizer'])

    # Extract embeddings
    net.eval()

    with torch.no_grad():  # set all 'requires_grad' to False
        for data in loader:
            if not multimodal:
                raise NotImplementedError

            else:
                omic1_test = data[0][0]
                omic2_test = data[1][0]

                # Encode test set in same encoder
                z1, z2 = net.encode(omic1_test, omic2_test)

                # Now decode data in different decoder
                omic1_from_omic2 = net.decoder(z2)
                omic2_from_omic1 = net.decoder2(z1)

                # Convert Tensors to numpy for evaluation
                if device != "cpu":
                    z1 = z1.cpu().numpy()
                    z2 = z2.cpu().numpy()
                    omic1_from_omic2 = omic1_from_omic2.cpu().numpy()
                    omic2_from_omic1 = omic2_from_omic1.cpu().numpy()
                    omic1_test = omic1_test.cpu().numpy()
                    omic2_test = omic2_test.cpu().numpy()
                else:
                    z1 = z1.numpy()
                    z2 = z2.numpy()
                    omic1_from_omic2 = omic1_from_omic2.numpy()
                    omic2_from_omic1 = omic2_from_omic1.numpy()
                    omic1_test = omic1_test.numpy()
                    omic2_test = omic2_test.numpy()

                # Imputation losses
                NR_MODALITIES = 2

                # mse[i,j]: performance of using modality i to predict modality j
                mse = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
                rsquared = np.eye(NR_MODALITIES)
                spearman = np.zeros((NR_MODALITIES, NR_MODALITIES, 2), float) # ,2 since we report mean and median
                spearman_p = np.zeros((NR_MODALITIES, NR_MODALITIES), float)

                # From x to y
                mse[0, 1], rsquared[0, 1], spearman[0, 1], spearman_p[0, 1] =\
                    evaluate_imputation(omic2_from_omic1, omic2_test, num_features2, 'mse'),\
                    evaluate_imputation(omic2_from_omic1, omic2_test, num_features2, 'rsquared'), \
                    evaluate_imputation(omic2_from_omic1, omic2_test, num_features2, 'spearman_corr'),\
                    evaluate_imputation(omic2_from_omic1, omic2_test, num_features2, 'spearman_p')
                mse[1, 0], rsquared[1, 0], spearman[1, 0], spearman_p[1, 0] =\
                    evaluate_imputation(omic1_from_omic2, omic1_test, num_features1, 'mse'),\
                    evaluate_imputation(omic1_from_omic2, omic1_test, num_features1, 'rsquared'), \
                    evaluate_imputation(omic1_from_omic2, omic1_test, num_features1, 'spearman_corr'), \
                    evaluate_imputation(omic1_from_omic2, omic1_test, num_features1, 'spearman_p')

                performance = {'mse': mse, 'rsquared': rsquared, 'spearman_corr': spearman, 'spearman_p': spearman_p}
                with open(save_dir + "/CGAE_task1_results.pkl", 'wb') as f:
                    pickle.dump(performance, f)

                logger.info("Performance: {}".format(performance))
                np.save("{}/task1_z1.npy".format(save_dir), z1)
                np.save("{}/task1_z2.npy".format(save_dir), z2)
                save_factorizations_to_csv(z1, sample_names, save_dir, 'task1_z1')
                save_factorizations_to_csv(z2, sample_names, save_dir, 'task1_z2')


    return z1, z2


def extract(net, model_file, loader, save_dir, multimodal=False):
    ## not 100% happy with the way embeddings are saved now
    ## we need separate directories for train/validation/test data
    checkpoint = torch.load(model_file)
    net.load_state_dict(checkpoint['state_dict'])
    net.opt.load_state_dict(checkpoint['optimizer'])

    # Extract embeddings
    net.eval()

    with torch.no_grad():  # set all 'requires_grad' to False
        for data in loader:
            if not multimodal:
                raise NotImplementedError

            else:
                ge_test = data[0][0]
                me_test = data[1][0]

                # Encode test set in same encoder
                z1, z2 = net.encode(ge_test, me_test)
                z1 = z1.cpu().numpy().squeeze()
                z2 = z2.cpu().numpy().squeeze()

                np.save("{}/task2_z1.npy".format(save_dir), z1)
                np.save("{}/task2_z2.npy".format(save_dir), z2)

    return z1, z2


def load_checkpoint(net, filename='model_last.pth.tar'):
    start_epoch = 0
    try:
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        net.opt.load_state_dict(checkpoint['optimizer'])

        print("\n[*] Loaded checkpoint at epoch %d" % start_epoch)
    except:
        print("[!] No checkpoint found, start epoch 0")

    return start_epoch
