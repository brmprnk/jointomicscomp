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


def run(args: dict) -> None:
    logger.success("Now starting MVIB")
    print(training_module)

    save_dir = os.path.join(args['save_dir'], '{}'.format('MVIB'))
    os.makedirs(save_dir)
    #
    # parser.add_argument("--data-dir", type=str, default='.', help="Root path for the datasets.")
    # parser.add_argument("--no-logging", action="store_true", help="Disable tensorboard logging")
    # parser.add_argument("--overwrite", action="store_true",
    # 					help="Force the over-writing of the previous experiment in the specified directory.")
    # parser.add_argument("--device", type=str, default="cuda:0",
    # 					help="Device on which the experiment is executed (as for tensor.device). Specify 'cpu' to "
    # 						 "force execution on CPU.")
    # parser.add_argument("--num-workers", type=int, default=8,
    # 					help="Number of CPU threads used during the data loading procedure.")
    # parser.add_argument("--batch-size", type=int, default=64, help="Batch size used for the experiments.")
    # parser.add_argument("--load-model-file", type=str, default=None,
    # 					help="Checkpoint to load for the experiments. Note that the specified configuration file needs "
    # 						 "to be compatible with the checkpoint.")
    # parser.add_argument("--checkpoint-every", type=int, default=50, help="Frequency of model checkpointing (in epochs).")
    # parser.add_argument("--backup-every", type=int, default=5, help="Frequency of model backups (in epochs).")
    # parser.add_argument("--evaluate-every", type=int, default=5, help="Frequency of model evaluation.")
    # parser.add_argument("--epochs", type=int, default=1000, help="Total number of training epochs")

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

    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, args['trainer'])
    trainer = TrainerClass(log_loss_every=len(train_loader), writer=tf_logger, **{
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



    ##########

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
            trainer.save(os.path.join(save_dir, 'checkpoint_%d.pt' % epoch))

    # Extract Phase #

    # Imputation
    if args['task'] == 1:
        logger.info("Imputation: Extracting Z1 and Z2 using test set")

        dataExtract1 = omic1[test_ind]
        dataExtract2 = omic2[test_ind]

        dataExtract1 = torch.tensor(dataExtract1, device=device)
        dataExtract2 = torch.tensor(dataExtract2, device=device)

        datasetExtract = MultiOmicsDataset(dataExtract1, dataExtract2, cancer_type_index[test_ind])

        train_loader = DataLoader(datasetExtract, batch_size=args['batch_size'], shuffle=False, num_workers=0)

        # Compute imputation loss

        with torch.no_grad():  # set all 'requires_grad' to False
            for data in train_loader:

                omic1_test = data[0]
                omic2_test = data[1]

                # Encode test set in same encoder
                z1 = trainer.encoder_v1(omic1_test)
                z2 = trainer.encoder_v2(omic2_test)

                np.save("{}/task1_z1.npy".format(save_dir), z1)
                np.save("{}/task1_z2.npy".format(save_dir), z2)

                labels = np.load(args['cancer_type_index']).astype(int)
                test_labels = cancertypes[[labels[test_ind]]]

                z1_plot = UMAPPlotter(z1, test_labels, "MVIB Z1: Task {} | {} & {} \n"
                                                       "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                                      .format(args['task'], args['data1'], args['data2'],
                                              args['epochs'], args['z_dim'], args['encoder_lr'], args['batch_size']),
                                      save_dir + "/{} UMAP.png".format('CGAE'))
                z1_plot.plot()

                z2_plot = UMAPPlotter(z2, test_labels, "MVIB Z2: Task {} | {} & {} \n"
                                                       "Epochs: {}, Latent Dimension: {}, LR: {}, Batch size: {}"
                                      .format(args['task'], args['data1'], args['data2'],
                                              args['epochs'], args['z_dim'], args['encoder_lr'], args['batch_size']),
                                      save_dir + "/{} UMAP.png".format('CGAE'))
                z2_plot.plot()

                # Now decode data in different decoder
                # ge_from_me = net.decoder(z2)
                # me_from_ge = net.decoder2(z1)
                #
                # imputation_loss_ge = mean_squared_error(ge_test, ge_from_me)
                # imputation_loss_me = mean_squared_error(me_test, me_from_ge)
                #
                # print("z1", imputation_loss_ge, "z2", imputation_loss_me)
                #
                # logger.info("Imputation Loss for Gene Expression: ".format(imputation_loss_ge))
                # logger.info("Imputation Loss for Methylation: ".format(imputation_loss_me))
                # np.save("{}/task1_z1.npy".format(save_dir), z1)
                # np.save("{}/task1_z2.npy".format(save_dir), z2)
