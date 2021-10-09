import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Load data
data_dir = "/home/bram/jointomicscomp/data/"

ge = np.load(data_dir + "GE.npy")
gcn = np.load(data_dir + "GCN.npy")
me = np.load(data_dir + "ME.npy")

sample_names = np.load(data_dir + "sampleNames.npy")
cancerTypes = np.load(data_dir + "cancerTypes.npy")
cancerType = np.load(data_dir + "cancerType.npy")

assert len(ge) == len(gcn) == len(me) == len(sample_names) == len(cancerType), "data has sample mismatches"

# Splitting is done on sample_names. We only need to save the indices of the split,
# so sample_names was chosen arbitrarily.

# First get test set (10%)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

sss1 = split1.split(sample_names, cancerType)

# sss1 is a generator object, so use this weird syntax
trainValidInd = 0
testInd = 0
for i, j in sss1:
    trainValidInd = i
    testInd = j

# Let X denote data and let y denote labels
Xtest = sample_names[testInd]

ytest = cancerType[testInd]

XtrainValid = sample_names[trainValidInd]

ytrainValid = cancerType[trainValidInd]

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
