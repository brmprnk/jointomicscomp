"""
Main file for running Multi-omics Factor Analysis V2.

Additional documentation found in https://biofam.github.io/MOFA2/tutorials.html
Model can be trained in Python, but as of this moment downstream analysis can only be done in R.
This file will therefore also run an R script, since the Z matrix needs to be fetched in order to calculate
reconstruction loss.
"""
import os
import pandas as pd
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
    train_mofa(args, output_file)

    # Do computations in R


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


def reconstruction_loss(args: dict) -> None:
    """
    Calculating the reconstruction loss using the trained model and input data

    @param args:         Dictionary containing input parameters
    @return: None
    """

    result_output_path = "/Users/bram/rp-group-21-bpronk/results/80split_recon_loss.npy"

    print("Reading original data...")
    omic_data1 = pd.read_csv(args['data_path1'], index_col=0)
    omic_data2 = pd.read_csv(args['data_path1'], index_col=0)

    trained_indices = np.load("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_MOFA_DATA_indices.npy")
    omic_data1 = omic_data1.iloc[trained_indices]
    omic_data2 = omic_data2.iloc[trained_indices]

    logger.info("Omic data 1 shape {}".format(omic_data1.shape))
    logger.info("Omic data 2 shape {}".format(omic_data2.shape))
    logger.success("Finished reading original data, now calculate reconstruction losses")

    # Now get the results from MOFA
    # W = 5000 x 10 : Factors on columns, Features on Rows
    # Z = 10 * 9992 : Samples on columns, Factors on rows

    # Y = WZ = 5000 * 9992 : Features on rows, Samples on columns

    W = pd.read_csv("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_W.csv")
    Z = pd.read_csv("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_Z.csv")

    # Get Z matrix
    unique_factors = np.unique(Z['factor'].values)

    z_matrix = []

    for factor in tqdm(range(len(unique_factors))):
        z_matrix.append((Z['value'].loc[Z['factor'] == unique_factors[factor]]).values)

    Z = np.matrix(z_matrix)

    # Get W matrix for each modality
    W_RNA = W.loc[W['view'] == "RNA-seq"]
    W_GCN = W.loc[W['view'] == "GENE CN"]
    W_DNA = W.loc[W['view'] == "DNA"]

    unique_features_rna = W_RNA['feature'].values[::10]
    matrix = []
    for i in tqdm(range(unique_features_rna.shape[0])):
        matrix.append(W_RNA['value'].loc[W_RNA['feature'] == unique_features_rna[i]])

    W_RNA = np.matrix(matrix)

    unique_features_gcn = W_GCN['feature'].values[::10]
    matrix = []
    for i in tqdm(range(unique_features_gcn.shape[0])):
        matrix.append(W_GCN['value'].loc[W_GCN['feature'] == unique_features_gcn[i]])

    W_GCN = np.matrix(matrix)

    unique_features_dna = W_DNA['feature'].values[::10]
    matrix = []
    for i in tqdm(range(unique_features_dna.shape[0])):
        matrix.append(W_DNA['value'].loc[W_DNA['feature'] == unique_features_dna[i]])

    W_DNA = np.matrix(matrix)

    print(W_RNA.shape, Z.shape)
    print(W_GCN.shape, Z.shape)
    print(W_DNA.shape, Z.shape)

    # Now get original values back (Y = WZ)
    Y_RNA = np.matmul(W_RNA, Z)
    Y_GCN = np.matmul(W_GCN, Z)
    Y_DNA = np.matmul(W_DNA, Z)

    # Get back in the form of original data
    Y_RNA = Y_RNA.transpose()
    Y_GCN = Y_GCN.transpose()
    Y_DNA = Y_DNA.transpose()

    print(Y_RNA)
    print(Y_RNA.shape)
    print(Y_GCN)
    print(Y_GCN.shape)
    print(Y_DNA)
    print(Y_DNA.shape)

    # input, predict
    rna_recon_loss = mean_squared_error(RNA_DATA.values, Y_RNA)
    print("RNA Reconstruction loss = ", rna_recon_loss)

    # input, predict
    gcn_recon_loss = mean_squared_error(GCN_DATA.values, Y_GCN)
    print("GCN Reconstruction loss = ", gcn_recon_loss)

    # input, predict
    dna_recon_loss = mean_squared_error(DNA_DATA.values, Y_GCN)
    print("DNA Reconstruction loss = ", dna_recon_loss)

    result = np.array([rna_recon_loss, gcn_recon_loss, dna_recon_loss])
    np.save(result_output_path, result)
