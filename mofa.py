import pandas as pd
import numpy as np
import sys
from mofapy2.run.entry_point import entry_point
import os

NUM_FACTORS = 10

############## SETUP INPUT/OUTPUT PATHS ##############

# input
INPUT_FILE_NAME = "80split_shuffle_MOFA_DATA.csv"

# output
OUTPUT_FILE_NAME = "80Split_Shuffledsamples_Trained_MOFA_{}Factors.hdf5".format(NUM_FACTORS)

# # Assumes data is found in the current working directory's /data folder
input_data_path = os.path.join(os.getcwd(), "data", INPUT_FILE_NAME)
output_data_path = os.path.join(os.getcwd(), "trained_models", OUTPUT_FILE_NAME)

######################################################


###########################
## Initialise MOFA model ##
###########################


## (1) initialise the entry point ##
ent = entry_point()


## (2) Set data options ##
# - scale_groups: if groups have significantly different ranges, it is good practice to scale each group to unit variance
# - scale_views: if views have significantly different ranges, it is good practice to scale each view to unit variance
ent.set_data_options(
	scale_groups = False, 
	scale_views = False
)

# samples_names nested list with length NGROUPS. Each entry g is a list with the sample names for the g-th group
# - if not provided, MOFA will fill it with default samples names
# samples_names = (...)

# features_names nested list with length NVIEWS. Each entry m is a list with the features names for the m-th view
# - if not provided, MOFA will fill it with default features names
# features_names = (...)

# ent.set_data_matrix(data, 
# 	views_names = views_names, 
# 	groups_names = groups_names, 
# 	samples_names = samples_names,   
# 	features_names = features_names
# )

# (3, option 2) Set data using a long data frame
print("MOFA DATA : Reading ...")
ent.set_data_df(pd.read_csv(input_data_path))
print("MOFA DATA : Loading Successful!")

## (4) Set model options ##
# - factors: number of factors. Default is K=10
# - likelihods: likelihoods per view (options are "gaussian","poisson","bernoulli"). 
# 		Default is None, and they are infered automatically
# - spikeslab_weights: use spike-slab sparsity prior in the weights? (recommended TRUE)
# - ard_factors: use ARD prior in the factors? (TRUE if using multiple groups)
# - ard_weights: use ARD prior in the weights? (TRUE if using multiple views)

# Simple (using default values)
ent.set_model_options()

# Advanced (using personalised values)
ent.set_model_options(
	factors = NUM_FACTORS, 
	spikeslab_weights = False, 
	ard_factors = False, 
	ard_weights = True
)


## (5) Set training options ##
# - iter: number of iterations
# - convergence_mode: "fast", "medium", "slow". 
#		For exploration, the fast mode is good enough.
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
	iter = 100, 
	convergence_mode = "medium", 
	startELBO = 1, 
	freqELBO = 1, 
	dropR2 = None, 
	gpu_mode = False, 
	verbose = True, 
	seed = 1
)


## (6, optional) Set stochastic inference options##
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

####################################
## Build and train the MOFA model ##
####################################

# Build the model 
ent.build()

# Run the model
ent.run()

##################################################################
## (Optional) do dimensionality reduction from the MOFA factors ##
##################################################################

# ent.umap()
# ent.tsne()

####################
## Save the model ##
####################

# - save_data: logical indicating whether to save the training data in the hdf5 file.
# this is useful for some downstream analysis in R, but it can take a lot of disk space.
ent.save(output_data_path, save_data=True)


