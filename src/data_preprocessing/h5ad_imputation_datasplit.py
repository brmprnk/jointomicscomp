import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

rna = np.load("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD.npy", allow_pickle=True)
if len(rna) == 0:  # Can be old relic from scipy formatting when loaded in after h5ad_get5000MAD.py
    print(rna)
    rna = rna.item().toarray()

# and store as proper file
np.save("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD.npy", rna.astype(np.float32))
protein = np.load("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD.npy", allow_pickle=True)
# plus store in right file format
np.save("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD.npy", protein.astype(np.float32))

data_dir = "/home/bram/jointomicscomp/data/CELL/task1/"

# get sample names from protein data (=equal to rna data)
sample_names = np.load("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_sampleNames.npy", allow_pickle=True).astype(str)

# get celltype.l2 from protein data (=equal to rna data)
cellType = np.load("/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_celltype_l2.npy", allow_pickle=True).astype(str)

assert len(rna) == len(protein) == len(sample_names) == len(cellType), "data has sample mismatches"

# Splitting is done on sample_names. We only need to save the indices of the split,
# so sample_names was chosen arbitrarily.

# First get test set (10%)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

sss1 = split1.split(sample_names, cellType)

# sss1 is a generator object, so use this weird syntax
trainValidInd = 0
testInd = 0
for i, j in sss1:
    trainValidInd = i
    testInd = j

# Let X denote data and let y denote labels
Xtest = sample_names[testInd]

ytest = cellType[testInd]

XtrainValid = sample_names[trainValidInd]

ytrainValid = cellType[trainValidInd]

# Now we get the validation set as 1/9 of the remaining 90%
split2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9)

sss2 = split2.split(XtrainValid, ytrainValid)

# sss2 is a generator object, so use this weird syntax
for i, j in sss2:
    trainInd = i
    validInd = j

np.save(data_dir + "trainInd.npy", trainInd)
np.save(data_dir + "validInd.npy", validInd)
np.save(data_dir + "testInd.npy", testInd)