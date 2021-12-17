"""
WARNING: Make sure to run h5ad_get5000MAD.py before running this file.
This file relies heavily on files created and stored in h5ad_get5000MAD.py.

File accomplishing three things
1 ) Makes sure the data matrices are stored as numpy data, instead of a sparse csr_matrix (so it can be used in torch).
2 ) Fetches sample names and celltype (l1, l2, l3) labels from the observations
3 ) Creates a training, validation and test split based on celltype.l2
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Path to the 5000MAD npy objects gathered in h5ad_get5000MAD.py
rna_path = "/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD.npy"
protein_path = "/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD.npy"

# Having saved the h5ad.X as a numpy object,
rna = np.load(rna_path, allow_pickle=True)
try:
    rna = np.array(rna.item().toarray(), dtype=np.float64)  # rna is now a np matrix
except ValueError as e:
    print(e)
    print("Tried to convert csr_matrix to numpy, but it was already a numpy matrix. Continuing progress")

# Overwrite the csr_matrix with the new numpy array
np.save(rna_path, rna)

# Now do the same conversion for protein data
protein = np.load(protein_path, allow_pickle=True)
try:
    protein = np.array(protein.item().toarray(), dtype=np.float64)  # rna is now a np matrix
except ValueError as e:
    print(e)
    print("Tried to convert csr_matrix to numpy, but it was already a numpy matrix. Continuing progress")

# plus store in right file format
np.save(protein_path, protein)

####################################################################################
# DATA IS NOW PREPROCESSED AND MADE TO BE NPY ARRAYS.
# CONTINUE WITH EXTRACTING INFORMATION FROM THE OBSERVATIONS, SUCH AS CELLTYPE,
# IN THE FORMAT observationType, observationTypes,
# WHERE observationTypes ARE THE UNIQUE VALUES OBSERVED, AND
# observationType IS AN ARRAY OF INDICES POINTING TO PLACES IN observationTypes.
####################################################################################

data_dir = "/home/bram/jointomicscomp/data/CELL/"

# Assumes obs .csv files have been created in h5ad_get5000MAD.py
rna_obs = pd.read_csv("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD_obs.csv", index_col=0)
protein_obs = pd.read_csv("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_obs.csv", index_col=0)

# Get and store sample names
rna_samples = rna_obs.index.values.astype(str)
protein_samples = protein_obs.index.values.astype(str)
assert np.array_equal(rna_samples, protein_samples), \
    "There is a mismatch between RNA and ADT sample names. Cannot continue further."

np.save(data_dir + "cell_sampleNames.npy", rna_samples)

# Now fetch and store columns we're interested in (celltypes)
for ct in ['celltype.l1', 'celltype.l2', 'celltype.l3']:
    cellTypes = np.unique(rna_obs[ct].values.astype(str))
    cellType = np.zeros(len(rna_samples)).astype(int)

    # This ensures the format of cellType having indices of values placed in cellTypes.
    for i in range(len(rna_obs[ct].values)):
        cellType[i] = np.where(cellTypes == rna_obs[ct].values[i])[0][0]

    np.save(data_dir + ct + 'cellTypes.npy', cellTypes)
    np.save(data_dir + ct + 'cellType.npy', cellType)


# Now create data splits for imputation (train, val, test)
# Splitting is done on celltype.l2
# In the split.split calls, rna_samples is passed too, as it is conveniently an array with the same len as the data.

# First get test set (10%)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

sss1 = split1.split(rna_samples, cellType)

# sss1 is a generator object, so use this weird syntax
trainValidInd = 0
testInd = 0
for i, j in sss1:
    trainValidInd = i
    testInd = j

# Let X denote data and let y denote labels
Xtest = rna_samples[testInd]

ytest = cellType[testInd]

XtrainValid = rna_samples[trainValidInd]

ytrainValid = cellType[trainValidInd]

# Now we get the validation set as 1/9 of the remaining 90%
split2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9)

sss2 = split2.split(XtrainValid, ytrainValid)

# sss2 is a generator object, so use this weird syntax
for i, j in sss2:
    trainInd = i
    validInd = j

# Save test splits
np.save(data_dir + "trainInd.npy", trainInd)
np.save(data_dir + "validInd.npy", validInd)
np.save(data_dir + "testInd.npy", testInd)

print("Successfully fetched data from h5ad and retrieved a split. Exiting script.")
