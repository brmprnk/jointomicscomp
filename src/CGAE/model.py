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


class MultiOmicsDataset():
    def __init__(self, data1, data2):
        self.d1 = TensorDataset(data1)
        # load the 2nd data view
        self.d2 = TensorDataset(data2)
        self.length = data1.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # get the index-th element of the 3 matrices
        return self.d1.__getitem__(index), self.d2.__getitem__(index)


def multiomic_collate(batch):
    d1 = [x[0] for x in batch]
    d2 = [x[1] for x in batch]
    print(len(d1))
    print(len(d1[0]))

    return torch.from_numpy(np.array(d1)), torch.from_numpy(np.array(d2))


def train(device, net, num_epochs, train_loader, train_loader_eval, valid_loader, ckpt_dir, logs_dir, early_stopping,
          save_step=10, multimodal=False):
    # Define logger
    tf_logger = SummaryWriter(logs_dir)

    # Load checkpoint model and optimizer
    start_epoch = load_checkpoint(net, filename=ckpt_dir + '/model_last.pth.tar')

    # Evaluate validation set before start training
    print("[*] Evaluating epoch %d..." % start_epoch)
    for trd in train_loader_eval:
        if not multimodal:
            x = trd[0]
            metrics = net.evaluate(x)
        else:
            x1 = trd[0][0]
            x2 = trd[1][0]
            metrics = net.evaluate(x1, x2)

        print(metrics.keys())
        assert 'loss' in metrics
        print("--- Training loss:\t%.4f" % metrics['loss'])

    for vld in valid_loader:
        if not multimodal:
            x = vld[0]
            metrics = net.evaluate(x)
        else:
            x1 = vld[0][0]
            x2 = vld[1][0]
            metrics = net.evaluate(x1, x2)

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
                    current_loss = net.compute_loss(data[0])
                else:
                    current_loss = net.compute_loss(data[0][0], data[1][0])

                # Backward pass and optimize
                current_loss.backward()
                net.opt.step()

        # Save last model
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer': net.opt.state_dict()}
        torch.save(state, ckpt_dir + '/model_last.pth.tar')

        # Save model at epoch
        if (epoch + 1) % save_step == 0:
            print("[*] Saving model epoch %d..." % (epoch + 1))
            torch.save(state, ckpt_dir + '/model_epoch%d.pth.tar' % (epoch + 1))

        # Evaluate all training set and validation set at epoch
        print("[*] Evaluating epoch %d..." % (epoch + 1))

        for data in train_loader_eval:
            if not multimodal:
                metricsTrain = net.evaluate(data[0])
            else:
                metricsTrain = net.evaluate(data[0][0], data[1][0])

        for data in valid_loader:
            if not multimodal:
                metricsValidation = net.evaluate(data[0])
            else:
                metricsValidation = net.evaluate(data[0][0], data[1][0])

        print("--- Training loss:\t%.4f" % metricsTrain['loss'])
        print("--- Validation loss:\t%.4f" % metricsValidation['loss'])

        early_stopping(metricsValidation['loss'])

        for m in metricsTrain:
            tf_logger.add_scalar(m + '/train', metricsTrain[m], epoch + 1)
            tf_logger.add_scalar(m + '/validation', metricsValidation[m], epoch + 1)

        # Stop training when not improving
        if early_stopping.early_stop:
            logger.info('Early stopping training since loss did not improve for {} epochs.'
                        .format(early_stopping.patience))
            break

    print("[*] Finish training.")


def impute(net, model_file, loader, save_dir, sample_names, num_features1, num_features2, multimodal=False):
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

                # Imputation losses
                NR_MODALITIES = 2

                # mse[i,j]: performance of using modality i to predict modality j
                mse = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
                rsquared = np.eye(NR_MODALITIES)
                spearman = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
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
                with open(save_dir + "/CGAE results_pickle", 'wb') as f:
                    pickle.dump(performance, f)

                logger.info("Performance: {}".format(performance))
                np.save("{}/task1_z1.npy".format(save_dir), z1)
                np.save("{}/task1_z2.npy".format(save_dir), z2)
                save_factorizations_to_csv(z1.numpy(), sample_names, save_dir, 'task1_z1')
                save_factorizations_to_csv(z2.numpy(), sample_names, save_dir, 'task1_z2')


    return z1, z2


def extract(net, model_file, loader, save_dir, multimodal=False):
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
