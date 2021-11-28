"""
Main file for running Multi-omics Factor Analysis V2.

Additional documentation found in https://biofam.github.io/MOFA2/tutorials.html
Model can be trained in Python, but as of this moment downstream analysis can only be done in R.
This file will therefore also run an R script, since the Z matrix needs to be fetched in order to calculate
reconstruction loss.
"""
import os
import sys

# R needs to be installed, and this path needs to be set to the R_Home folder, found by running R.home() in R console.
os.environ['R_HOME'] = "/usr/lib/R"
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np
import gc
import pickle
from mofapy2.run.entry_point import entry_point
from src.util import logger
from src.util.umapplotter import UMAPPlotter
from src.util.evaluate import evaluate_imputation


def run(args: dict) -> None:
    """
    Setup before running MOFA+

    @param args: Dictionary containing input parameters
    @return: None
    """
    logger.info("Running MOFA+...")

    # Setup output paths
    save_dir = os.path.join(args['save_dir'], 'MOFA+')
    os.makedirs(save_dir)
    output_file = os.path.join(save_dir, "{}_{}_trained_MOFA_model.hdf5".format(args['data1'], args['data2']))

    # Run MOFA+
    if args['pre_trained'] == "":
        train_mofa(args, output_file)
    else:
        logger.info("Using previously trained model from : {}".format(args['pre_trained']))
        output_file = args['pre_trained']

    # Do computations in R
    try:
        get_W_and_Z(save_dir, output_file)
    except Exception as e:
        logger.error("{}".format(e))
        logger.error("This implementation tries to run R scripts in Python using rpy2 to get W and Z from the model."
                     "This requires setup of the $Home folder in the mofa.py file."
                     "It is not a required step, and can be ran manually in mofa_downstream.R")
        if args['pre_trained'] == "":
            logger.error("The Trained Model has been saved ")

    # Calculate task1 or task2
    downstream_analysis(args, save_dir)


def train_mofa(args: dict, save_file_path: str) -> None:
    """
    Run Multi-omics Factor Analysis V2

    @param args:           Dictionary containing input parameters
    @param save_file_path: Name of trained model that can be saved in /results dir
    @return: None
    """
    #########################
    # Initialise MOFA model #
    #########################

    # (1) initialise the entry point ##
    ent = entry_point()

    # (2) Set data options ##
    # - scale_groups: if groups have significantly different ranges, good practice to scale each group to unit variance
    # - scale_views: if views have significantly different ranges, its good practice to scale each view to unit variance
    ent.set_data_options(
        scale_groups=args['scale_groups'],
        scale_views=args['scale_views']
    )

    # samples_names nested list with length NGROUPS. Each entry g is a list with the sample names for the g-th group

    # (3) Set data using a long data frame
    logger.info("MOFA DATA : Reading ...")
    # Load in data, depending on task
    # Task 1 : Imputation
    if args['task'] == 1:
        logger.info("Running Task {} on omic {} and omic {}".format(args['task'], args['data1'], args['data2']))

        # Load in data
        omic1 = np.load(args['data_path1']).astype(np.float32)
        omic2 = np.load(args['data_path2']).astype(np.float32)
        sample_names = np.load(args['sample_names']).astype(str)

        # Use predefined split
        train_ind = np.load(args['train_ind'])
        val_ind = np.load(args['val_ind'])

        # Train on training + validation set for fair comparison

        omic1 = np.concatenate((omic1[train_ind], omic1[val_ind]))
        omic2 = np.concatenate((omic2[train_ind], omic2[val_ind]))
        sample_names = np.concatenate((sample_names[train_ind], sample_names[val_ind]))

        print("rewrote omics")

        data_mat = np.zeros((2, 1), dtype=object)

        data_mat[0][0] = omic1
        data_mat[1][0] = omic2
        data_mat = list(data_mat)


        # MOFA+ Wants all your memory.
        # Clean this data loaded, as it has been stored in data_mat
        del omic1
        del omic2

        gc.collect()

        try:
            ent.set_data_matrix(data_mat, likelihoods=["gaussian", "gaussian"], views_names=[args['data1'], args['data2']],
                                features_names=[np.load(args['data_features1'], allow_pickle=True).astype(str).tolist(),
                                                np.load(args['data_features2'], allow_pickle=True).astype(str).tolist()],
                                samples_names=[sample_names])
        except AssertionError as e:
            print(e)

    logger.success("MOFA DATA : Loading Successful!")

    # (4) Set model options ##
    # - factors: number of factors. Default is K=10
    # - likelihoods: likelihoods per view (options are "gaussian","poisson","bernoulli").
    # 		Default is None, and they are infered automatically
    # - spikeslab_weights: use spike-slab sparsity prior in the weights? (recommended TRUE)
    # - ard_factors: use ARD prior in the factors? (TRUE if using multiple groups)
    # - ard_weights: use ARD prior in the weights? (TRUE if using multiple views)

    # Simple (using default values)
    ent.set_model_options()

    # Advanced (using personalised values)
    ent.set_model_options(
        factors=args['factors'],
        spikeslab_weights=args['spikeslab_weights'],
        ard_factors=args['ard_factors'],
        ard_weights=args['ard_weights']
    )

    # (5) Set training options ##
    # - iter: number of iterations
    # - convergence_mode: "fast", "medium", "slow".
    #   For exploration, the fast mode is good enough.
    # - startELBO: initial iteration to compute the ELBO (the objective function used to assess convergence)
    # - freqELBO: frequency of computations of the ELBO (the objective function used to assess convergence)
    # - dropR2: minimum variance explained criteria to drop factors while training.
    # 		Default is None, inactive factors are not dropped during training
    # - gpu_mode: use GPU mode? this needs cupy installed and a functional GPU, see https://cupy.chainer.org/
    # - verbose: verbose mode?
    # - seed: random seed

    # Simple (using default values)
    ent.set_train_options()

    # Advanced (using personalised values)
    ent.set_train_options(
        iter=args['iterations'],
        convergence_mode=args['convergence_mode'],
        startELBO=args['startELBO'],
        freqELBO=args['freqELBO'],
        dropR2=args['dropR2'],
        gpu_mode=args['cuda'],
        verbose=args['verbose'],
        seed=args['random_seed']
    )

    # (6, optional) Set stochastic inference options##
    # Only recommended with very large sample size (>1e6) and when having access to GPUs
    # - batch_size: float value indicating the batch size (as a fraction of the total data set: 0.10, 0.25 or 0.50)
    # - learning_rate: learning rate (we recommend values from 0.25 to 0.75)
    # - forgetting_rate: forgetting rate (we recommend values from 0.25 to 0.5)
    # - start_stochastic: first iteration to apply stochastic inference (recommended > 5)

    # Simple (using default values)
    # ent.set_stochastic_options()

    # Advanced (using personalised values)
    # ent.set_stochastic_options(batch_size=0.5, learning_rate=0.75, forgetting_rate=0.5, start_stochastic=10)

    # ent.set_stochastic_options(
    #     batch_size = 0.1
    # )

    ##################################
    # Build and train the MOFA model #
    ##################################

    # Build the model
    ent.build()

    # Run the model
    ent.run()

    # - save_data: logical indicating whether to save the training data in the hdf5 file.
    # this is useful for some downstream analysis in R, but it can take a lot of disk space.
    ent.save(save_file_path, save_data=args['save_data'])
    logger.success("Trained MOFA+ model has been saved to {}".format(save_file_path))


def get_W_and_Z(save_dir: str, model_file: str) -> None:
    """
    Perform R based code here in Python.
    Documentation: https://rpy2.github.io/doc/v2.9.x/html/introduction.html

    @param save_dir:   path to directory where factors and weights should be saved
    @param model_file: path to trained model

    @return: None
    """
    # import R's "base" package
    rpackages.importr('base')

    # import R's "utils" package
    utils = rpackages.importr('utils')

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

    # MOFA required R package names
    package_names = ('ggplot2', 'MOFA2')

    # R vector of strings
    from rpy2.robjects.vectors import StrVector

    # Selectively install what needs to be installed
    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    robjects.r('''
            library(ggplot2)
            library(MOFA2)
            # create a function `f` that saves model factors and weights
            save_z <- function(save_dir, model_path, verbose=FALSE) {
                trained_model <- (model_path)

                model <- load_model(trained_model, remove_inactive_factors = F)
                
                Z = get_expectations(model, "Z", as.data.frame = TRUE)
                W = get_expectations(model, "W", as.data.frame = TRUE)
                write.csv(Z, paste(save_dir, "Z.csv", sep="/"), row.names = FALSE)
                write.csv(W, paste(save_dir, "W.csv", sep="/"), row.names = FALSE)
            }
            ''')

    r_save_z = robjects.globalenv['save_z']

    # Save model factors and weights (Z and W)
    r_save_z(save_dir, model_file)


def downstream_analysis(args: dict, save_dir: str) -> None:
    """
    Calculating the reconstruction loss using the trained model and input data

    @param args:         Dictionary containing input parameters
    @param save_dir:     path to directory where factors and weights should be saved

    @return: None
    """
    logger.info("Reading original data...")
    if args['task'] == 1:
        omic_data1 = np.load(args['data_path1'])
        omic_data2 = np.load(args['data_path2'])

        test_ind = np.load(args['test_ind'])
        omic_data1 = omic_data1[test_ind]
        omic_data2 = omic_data2[test_ind]

        logger.info("Omic data 1 test set shape {}".format(omic_data1.shape))
        logger.info("Omic data 2 test set shape {}".format(omic_data2.shape))
    else:
        omic_data1 = np.load(args['x_ctype_test_file'])
        omic_data2 = np.load(args['y_ctype_test_file'])

        logger.info("Omic data 1 test set {} shape {}".format(args['ctype'], omic_data1.shape))
        logger.info("Omic data 2 test set {} shape {}".format(args['ctype'], omic_data2.shape))

    logger.success("Finished reading original data, now calculate reconstruction losses")

    # Now get the results from MOFA
    logger.info("Fetching MOFA+ Z and W")
    logger.info("These should be placed in the directory of the trained model as W.csv and Z.csv ")

    if args['pre_trained'] == "":
        W = pd.read_csv(os.path.join(save_dir, "W.csv"))
        Z = pd.read_csv(os.path.join(save_dir, "Z.csv"))
    else:
        W = pd.read_csv(os.path.join(os.path.dirname(args['pre_trained']), "W.csv"))
        Z = pd.read_csv(os.path.join(os.path.dirname(args['pre_trained']), "Z.csv"))

    logger.info("W has shape {}".format(W.shape))
    logger.info("Z has shape {}".format(Z.shape))

    # Get Z matrix
    unique_factors = np.unique(Z['factor'].values)
    z_matrix = []

    for factor in tqdm(range(len(unique_factors))):
        z_matrix.append((Z['value'].loc[Z['factor'] == unique_factors[factor]]).values)

    Z = np.matrix(z_matrix)

    # Plot this total Z using UMAP
    np.save("{}/task1_z.npy".format(save_dir), z_matrix)

    labels = np.load(args['labels']).astype(int)
    labeltypes = np.load(args['labelnames']).astype(str)

    # Get correct labels with names
    print(labeltypes)
    print("That was labeltypes")
    print(labels)
    training_labels = np.concatenate((labeltypes[[labels[np.load(args['train_ind'])]]], labeltypes[[labels[np.load(args['val_ind'])]]]))

    training_data_plot = UMAPPlotter(Z.transpose(),
                                     training_labels,
                                     "MOFA+: Task {} Training Data's Z | {} & {} \nFactors: {}, Views: 2, Groups: 1"
                                     .format(args['task'], args['data1'], args['data2'], args['factors']),
                                     save_dir + "/MOFA+ UMAP.png")

    training_data_plot.plot()

    # Get W matrix for each modality
    try:
        W_omic1 = W.loc[W['view'] == args['data1']]
        W_omic2 = W.loc[W['view'] == args['data2']]
    except (ValueError, Exception) as e:
        logger.error(e)
        logger.error("Probably setup the wrong view names in MOFA+ reconstruction loss.")

    unique_features = W_omic1['feature'].values[:args['num_features1']]
    np.set_printoptions(threshold=sys.maxsize)

    # Turn W dataframes into np arrays for calculations
    matrix = np.zeros((len(unique_features), len(unique_factors)))
    for i in tqdm(range(args['num_features1'])):
        for j in range(args['factors']):
            matrix[i][j] = W_omic1['value'].values[j * args['num_features1'] + i]

    W_omic1 = matrix

    unique_features = W_omic2['feature'].values[:args['num_features2']]
    matrix = np.zeros((len(unique_features), len(unique_factors)))
    for i in tqdm(range(args['num_features2'])):
        for j in range(args['factors']):
            matrix[i][j] = W_omic2['value'].values[j * args['num_features2'] + i]

    W_omic2 = matrix

    # Imputation
    if args['task'] == 1:
        logger.info("Now do imputations using pseudo-inverse of W")

        # Imputation from Y1 to Y2
        W_pseudo1 = np.linalg.pinv(W_omic1)  # Shape (factors, features)

        Y1 = omic_data1.transpose()
        Z_frompseudo1 = np.matmul(W_pseudo1, Y1)

        print("W_pseudo1.shape : ", W_pseudo1.shape)
        print("And test set shape : ", Y1.shape)

        print("Z from W^+ * Y1: ", Z_frompseudo1.shape)
        np.save("{}/task1_z_from_pseudoinv_w1.npy".format(save_dir), Z_frompseudo1)

        # Now to impute Y2 from new Z
        Y2_impute = np.matmul(W_omic2, Z_frompseudo1)

        # Imputation from Y1 to Y2
        W_pseudo2 = np.linalg.pinv(W_omic2)  # Shape (factors, features)

        Y2 = omic_data2.transpose()
        Z_frompseudo2 = np.matmul(W_pseudo2, Y2)
        np.save("{}/task1_z_from_pseudoinv_w2.npy".format(save_dir), Z_frompseudo2)

        # Now to impute Y1 from new Z
        Y1_impute = np.matmul(W_omic1, Z_frompseudo2)

        # mse[i,j]: performance of using modality i to predict modality j
        mse = np.zeros((2, 2), float)
        rsquared = np.eye(2)

        mse[0, 1], rsquared[0, 1] = evaluate_imputation(omic_data2.transpose(), Y2_impute, 'mse'), evaluate_imputation(omic_data2.transpose(), Y2_impute, 'rsquared')
        mse[1, 0], rsquared[1, 0] = evaluate_imputation(omic_data1.transpose(), Y1_impute, 'mse'), evaluate_imputation(omic_data1.transpose(), Y1_impute, 'rsquared')

        performance = {'mse': mse, 'rsquared': rsquared}
        logger.info("{}".format(performance))
        with open(os.path.join(save_dir, args['name'] + 'results_pickle'), 'wb') as f:
            pickle.dump(performance, f)

        logger.info("Imputation loss {} from {} = {}".format(args['data1'], args['data2'], mse[0, 1]))
        logger.info("Imputation loss {} from {} = {}".format(args['data2'], args['data1'], mse[1, 0]))

        test_labels = labeltypes[[labels[test_ind]]]

        z1_plot = UMAPPlotter(Z_frompseudo1.transpose(),
                              test_labels,
                              "MOFA+: Task {} Z from $W^⁺Y_1$ | {} & {} \nFactors: {}, Views: 2, Groups: 1"
                              .format(args['task'], args['data1'], args['data2'], args['factors']),
                              save_dir + "/MOFA+ UMAP_Z_from_pseudoinv_w1.png")

        z1_plot.plot()

        z2_plot = UMAPPlotter(Z_frompseudo2.transpose(),
                              test_labels,
                              "MOFA+: Task {} Z from $W⁺Y_2$ | {} & {} \nFactors: {}, Views: 2, Groups: 1"
                              .format(args['task'], args['data1'], args['data2'], args['factors']),
                              save_dir + "/MOFA+ UMAP_Z_from_pseudoinv_w2.png")

        z2_plot.plot()

    # Prediction
    if args['task'] == 2:
        logger.info("Task 2 : Now fetch Z for each omic")

        # Imputation from Y1 to Y2
        W_pseudo1 = np.linalg.pinv(W_omic1)  # Shape (factors, features)

        Y1 = omic_data1.transpose()
        Z_frompseudo1 = np.matmul(W_pseudo1, Y1)

        print("W_pseudo1.shape : ", W_pseudo1.shape)
        print("And test set shape : ", Y1.shape)

        print("Z from W^+ * Y1: ", Z_frompseudo1.shape)
        np.save("{}/task2_z_from_pseudoinv_w1.npy".format(save_dir), Z_frompseudo1)

        W_pseudo2 = np.linalg.pinv(W_omic2)  # Shape (factors, features)

        Y2 = omic_data2.transpose()
        Z_frompseudo2 = np.matmul(W_pseudo2, Y2)
        np.save("{}/task2_z_from_pseudoinv_w2.npy".format(save_dir), Z_frompseudo2)

        logger.info("Z's are saved for later predictions.")



