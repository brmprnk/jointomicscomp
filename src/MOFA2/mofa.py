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
from mofapy2.run.entry_point import entry_point
from src.util import logger



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
    output_file = os.path.join(save_dir, "trained_MOFA_model.hdf5")

    # Run MOFA+
    if args['pre_trained'] == "":
        train_mofa(args, output_file)
    else:
        logger.info("Using previously trained model from : {}".format(args['pre_trained']))
        output_file = args['pre_trained']

    # Do computations in R
    downstream_mofa(save_dir, output_file)

    # Calculate reconstruction loss
    reconstruction_loss(args, save_dir)


def train_mofa(args: dict, output_model: str) -> None:
    """
    Run Multi-omics Factor Analysis V2

    @param args:         Dictionary containing input parameters
    @param output_model: Name of trained model that can be saved in /results dir
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
    ent.set_data_df(pd.read_csv(args['mofa_data_path']))
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
    ent.save(output_model, save_data=args['save_data'])


def downstream_mofa(save_dir: str, model_file: str) -> None:
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


def reconstruction_loss(args: dict, save_dir: str) -> None:
    """
    Calculating the reconstruction loss using the trained model and input data

    @param args:         Dictionary containing input parameters
    @param save_dir:   path to directory where factors and weights should be saved

    @return: None
    """

    global W_omic1, W_omic2

    logger.info("Reading original data...")
    omic_data1 = np.load(args['data_path1'])
    omic_data2 = np.load(args['data_path2'])

    logger.info("Omic data 1 shape {}".format(omic_data1.shape))
    logger.info("Omic data 2 shape {}".format(omic_data2.shape))
    logger.success("Finished reading original data, now calculate reconstruction losses")

    # Now get the results from MOFA
    # W = 5000 x 10 : Factors on columns, Features on Rows
    # Z = 10 * 9992 : Samples on columns, Factors on rows

    # Y = WZ = 5000 * 9992 : Features on rows, Samples on columns

    logger.info("Fetching MOFA+ Z and W")
    W = pd.read_csv(os.path.join(save_dir, "W.csv"))
    Z = pd.read_csv(os.path.join(save_dir, "Z.csv"))

    print("W = ", W)

    print("Z = ", Z)

    # Get Z matrix
    unique_factors = np.unique(Z['factor'].values)

    print("Unique factors = ", unique_factors)

    z_matrix = []

    for factor in tqdm(range(len(unique_factors))):
        z_matrix.append((Z['value'].loc[Z['factor'] == unique_factors[factor]]).values)

    Z = np.matrix(z_matrix)

    print("Z", Z)
    print(Z.shape)

    # Get W matrix for each modality
    try:
        W_omic1 = W.loc[W['view'] == "GE"]
        W_omic2 = W.loc[W['view'] == "ME"]
    except (ValueError, Exception) as e:
        logger.error(e)
        logger.error("Probably setup the wrong view names in MOFA+ reconstruction loss.")

    print(W_omic1)
    print("Feature, ", W_omic1['feature'])
    print("FEature values , ",  W_omic1['feature'].values)
    unique_features = W_omic1['feature'].values[:args['num_features']]
    print("len unique features", len(unique_features))
    np.set_printoptions(threshold=sys.maxsize)

    matrix = []
    for i in tqdm(range(unique_features.shape[0])):
        matrix.append(W_omic1['value'].loc[W_omic1['feature'] == unique_features[i]])

    W_omic1 = np.matrix(matrix)

    unique_features = W_omic2['feature'].values[:args['num_features']]
    matrix = []
    for i in tqdm(range(unique_features.shape[0])):
        matrix.append(W_omic2['value'].loc[W_omic2['feature'] == unique_features[i]])

    W_omic2 = np.matrix(matrix)

    print(W_omic1.shape, Z.shape)
    print(W_omic2.shape, Z.shape)

    # Now get original values back (Y = WZ)
    Y_omic1 = np.matmul(W_omic1, Z)
    Y_omic2 = np.matmul(W_omic2, Z)

    # Get back in the form of original data
    Y_omic1 = Y_omic1.transpose()
    Y_omic2 = Y_omic2.transpose()

    logger.info("Reconstructed data for omic 1 shape: {}".format(Y_omic1.shape))

    logger.info("Reconstructed data for omic 2 shape: {}".format(Y_omic2.shape))

    # input, predict
    ge_recon_loss = mean_squared_error(omic_data1, Y_omic1)

    # input, predict
    me_recon_loss = mean_squared_error(omic_data2, Y_omic2)

    result = np.array([ge_recon_loss, me_recon_loss])

    logger.info("Reconstruction loss GE = {}".format(ge_recon_loss))
    logger.info("Reconstruction loss ME = {}".format(me_recon_loss))

    np.save(os.path.join(save_dir, "recon_loss.npy"), result)

    logger.info("Now do imputations : Yi^T * Wj for all i = 1, 2 and j = 1, 2")

    Y_ge_W_me = np.matmul(omic_data1, W_omic2)
    Y_me_W_ge = np.matmul(omic_data2, W_omic1)

    print(Y_me_W_ge.shape)

    impute_Y_ge_W_me = mean_squared_error(Y_ge_W_me, np.transpose(Z))
    impute_Y_me_W_ge = mean_squared_error(Y_me_W_ge, np.transpose(Z))

    logger.info("Impute Y_GE and W_ME = {}".format(impute_Y_ge_W_me))
    logger.info("Impute Y_ME and W_GE = {}".format(impute_Y_me_W_ge))

    np.save(os.path.join(save_dir, "impute_loss.npy"), result)
    logger.success("Saved results. Exiting program.")