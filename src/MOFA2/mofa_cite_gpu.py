from mofapy2.run.entry_point import entry_point
import pandas as pd
import numpy as np
import sys

factors = int(sys.argv[1])

train_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/trainInd.npy')

rna_features = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/featureNamesRNA.npy', allow_pickle=True)
adt_features = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/featureNamesADT.npy', allow_pickle=True)
rna = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/RNA.npy')
adt = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/ADT.npy')
sample_names = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/sampleNames.npy', allow_pickle=True)

x1 = np.log(1 + rna[train_ind,:])
x2 = np.log(1 + adt[train_ind,:])

data = [None, None]
data[0] = [None]
data[1] = [None]
data[0][0] = x1
data[1][0] = x2


# initialise
ent = entry_point()

# Set data options
ent.set_data_options()

# Set data
ent.set_data_matrix(data)
#ent.set_data_df(data)

# Set model options
ent.set_model_options(factors=factors, spikeslab_weights=True)

# Set training options
ent.set_train_options(iter=500, freqELBO=5, dropR2=None, startELBO=1, verbose=False, seed=1, convergence_mode="medium", gpu_mode=1, save_interrupted=True)

# Build the model
ent.build()

# Train the model
ent.run()

ent.save('src/MOFA2/cite-model-gpu%d' % factors)

