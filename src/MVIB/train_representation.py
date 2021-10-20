import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.MVIB import training as training_module
from src.MVIB.utils.data import MultiOmicsDataset
from src.util import logger
from src.util.umapplotter import UMAPPlotter
from tensorboardX import SummaryWriter
from src.util.early_stopping import EarlyStopping


def run(args: dict) -> None:
    logger.success("Now starting MVIB")

    # Create save directory
    save_dir = os.path.join(args['save_dir'], '{}'.format('MVIB'))
    os.makedirs(save_dir)

    # Create directories for checkpoint, sample and logs files
    ckpt_dir = save_dir + '/checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logs_dir = save_dir + '/logs'

    tf_logger = SummaryWriter(save_dir)
    overwrite = args['overwrite']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args['pre_trained'] != "" and not overwrite:
        raise Exception("The experiment directory %s already contains a trained model, please specify a different "
                        "experiment directory. Going on will resume training with the parameters found"
                        "in the config file."
                        "or use the --overwrite flag to force overwriting")

    resume_training = args['pre_trained'] != "" and not overwrite

    # Load in data, depending on task
    # Task 1 : Imputation
    if args['task'] == 1:
        logger.info("Running Task {} on omic {} and omic {}".format(args['task'], args['data1'], args['data2']))

        # Load in data
        omic1 = np.load(args['data_path1'])
        omic2 = np.load(args['data_path2'])
        sample_names = np.load(args['sample_names'])
        cancertypes = np.load(args['cancertypes'])
        cancer_type_index = np.load(args['cancer_type_index'])

        # Use predefined split
        train_ind = np.load(args['train_ind'])
        val_ind = np.load(args['val_ind'])
        test_ind = np.load(args['test_ind'])

        mv_train_set = MultiOmicsDataset(omic1[train_ind], omic2[train_ind], cancer_type_index[train_ind])

        mv_val_set = MultiOmicsDataset(omic1[val_ind], omic2[val_ind], cancer_type_index[val_ind])

        # Initialization of the data loader
        train_loader = DataLoader(mv_train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)

        # Select a subset 100 samples (10 for each per label)
        # train_subset = split(train_set, 100, 'Balanced')

    # Setup early stopping, terminates training when validation loss does not improve for early_stopping_patience epochs
    early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)
    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, args['trainer'])
    trainer = TrainerClass(log_loss_every=len(train_loader), writer=tf_logger,
                           **{
                               'z_dim': args['z_dim'],
                               'input_dim1': args['input_dim1'],
                               'input_dim2': args['input_dim2'],
                               'optimizer_name': args['optimizer_name'],
                               'encoder_lr': args['encoder_lr'],
                               'miest_lr': args['miest_lr'],
                               'beta_start_value': args['beta_start_value'],
                               'beta_end_value': args['beta_end_value'],
                               'beta_n_iterations': args['beta_n_iterations'],
                               'beta_start_iteration': args['beta_start_iteration']
                           })

    # Resume the training if specified
    if resume_training:
        trainer.load(args['pre_trained'])

    # Moving the models to the specified device
    trainer.to(device)
    trainer.double()

    # Begin training loop
    for epoch in tqdm(range(args['epochs'])):
        for data in tqdm(train_loader):
            trainer.train_step(data)

        # Compute train and test_accuracy of a logistic regression
        # train_accuracy, test_accuracy = evaluate(encoder1=trainer.encoder_v1,
        #                                          encoder2=trainer.encoder_v2,
        #                                          train_on=mv_train_set,
        #                                          test_on=mv_val_set,
        #                                          device=device)
        # if not (writer is None):
        #     writer.add_scalar(tag='evaluation/train_accuracy', scalar_value=train_accuracy,
        #                       global_step=trainer.iterations)
        #     writer.add_scalar(tag='evaluation/test_accuracy', scalar_value=test_accuracy,
        #                       global_step=trainer.iterations)
        #
        # tqdm.write('Epoch {} Train Accuracy: {}'.format(epoch, train_accuracy))
        # tqdm.write('Epoch {} Test Accuracy: {}'.format(epoch, test_accuracy))

        if epoch % args['log_save_interval'] == 0:
            tqdm.write('Storing model checkpoint')
            trainer.save(os.path.join(ckpt_dir, 'checkpoint_%d.pt' % epoch))

        # Check for early stopping
        early_stopping(np.mean(trainer.loss_items['loss/total_L']))

        print("Epoch {}: Total loss = {}\n".format(epoch, trainer.loss_items['loss/total_L'][0]))

        # Stop training when not improving
        if early_stopping.early_stop:
            logger.info('Early stopping training since loss did not improve for {} epochs.'
                        .format(trainer.early_stop.patience))
            break

    # Extract Phase #

    # Imputation (not clearly defined since there is no decoder, simply save the two encoders' Z)
    if args['task'] == 1:
        logger.info("MVIB has no imputation: Extracting Z1 and Z2 using test set")

        dataExtract1 = omic1[test_ind]
        dataExtract2 = omic2[test_ind]

        dataExtract1 = torch.tensor(dataExtract1, device=device)
        dataExtract2 = torch.tensor(dataExtract2, device=device)

        datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2, cancer_type_index[test_ind])

        # Use 1 batch
        test_data_loader = DataLoader(datasetExtract, batch_size=len(dataExtract1), shuffle=False, num_workers=0)

        # Compute imputation loss

        with torch.no_grad():  # set all 'requires_grad' to False
            for data in test_data_loader:
                omic1_test = data[0]
                omic2_test = data[1]

                # Encode test set in same encoder
                z1 = trainer.encoder_v1(omic1_test).mean.numpy()
                z2 = trainer.encoder_v2(omic2_test).mean.numpy()

                np.save("{}/task1_z1.npy".format(save_dir), z1)
                np.save("{}/task1_z2.npy".format(save_dir), z2)

                labels = np.load(args['cancer_type_index']).astype(int)
                test_labels = cancertypes[[labels[test_ind]]]

                z1_plot = UMAPPlotter(z1, test_labels, "MVIB Z1: Task {} | {} & {} \n"
                                                       "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                                      .format(args['task'], args['data1'], args['data2'],
                                              args['epochs'], args['z_dim'], args['encoder_lr'], args['batch_size']),
                                      save_dir + "/MVIB Z1 UMAP.png")
                z1_plot.plot()

                z2_plot = UMAPPlotter(z2, test_labels, "MVIB Z2: Task {} | {} & {} \n"
                                                       "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                                      .format(args['task'], args['data1'], args['data2'],
                                              args['epochs'], args['z_dim'], args['encoder_lr'], args['batch_size']),
                                      save_dir + "/MVIB Z2 UMAP.png")
                z2_plot.plot()
