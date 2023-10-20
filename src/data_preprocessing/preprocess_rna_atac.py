import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

np.random.seed(12)
# https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis.html#wnn-analysis-of-10x-multiome-rna-atac
# ^ first follow these steps to get variable genes etc

rna = pd.read_table('data/scatac/countsRNA3kvar.tsv', header=0, index_col=0)
rna.columns = pd.Series(rna.columns).apply(lambda x: x.replace('.', '-'))

atac = pd.read_table('data/scatac/countsATAC5kvar.tsv', header=0, index_col=0)
atac.columns = pd.Series(atac.columns).apply(lambda x: x.replace('.', '-'))



y = pd.read_csv('atac/annotations.csv', header=0, index_col=0)

assert (atac.columns == y.index).all()
assert (rna.columns == y.index).all()

sampleNames = np.array(y.index)

rnaFeatureNames = np.array(rna.index)
atacFeatureNames = np.array(atac.index)

X1 = np.array(rna).T
X2 = np.array(atac).T


X2 = (X2 > 0).astype(float)


N = X1.shape[0]

labelsLevel1 = np.sort(y['Celltype'].unique())
l1toNo = dict()

for i, l in enumerate(labelsLevel1):
	l1toNo[l] = i

yLevel1 = np.array(y['Celltype'].map(l1toNo))



labelsLevel2 = np.sort(y['Subcelltype'].unique())
l2toNo = dict()

for i, l in enumerate(labelsLevel2):
	l2toNo[l] = i

yLevel2 = np.array(y['Subcelltype'].map(l2toNo))




sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

for trainBig_index, test_index in sss.split(X1, yLevel2):
	pass


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.11111111, random_state=0)

for trainInner_index, testInner_index in sss.split(X1[trainBig_index], yLevel2[trainBig_index]):
	pass

valInd = trainBig_index[testInner_index]
trainInd = trainBig_index[trainInner_index]


np.save('data/scatac/trainInd.npy', trainInd)
np.save('data/scatac/validInd.npy', valInd)
np.save('data/scatac/testInd.npy', test_index)


np.save('data/scatac/celltypes.npy', labelsLevel1)
np.save('data/scatac/celltypes_l2.npy', labelsLevel2)

np.save('data/scatac/celltype.npy', yLevel1)
np.save('data/scatac/celltype_l2.npy', yLevel2)


np.save('data/scatac/sampleNames.npy', sampleNames)
np.save('data/scatac/featureNamesRNA.npy', rnaFeatureNames)
np.save('data/scatac/featureNamesATAC.npy', atacFeatureNames)

np.save('data/scatac/rna.npy', X1)
np.save('data/scatac/atac.npy', X2)


dfRNA = pd.DataFrame(data=X1, index=sampleNames, columns=rnaFeatureNames)
dfATAC = pd.DataFrame(data=X2, index=sampleNames, columns=atacFeatureNames)

dfRNA.iloc[trainInd].to_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/atac_rna_train.csv')
dfRNA.iloc[valInd].to_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/atac_rna_valid.csv')
dfRNA.iloc[test_index].to_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/atac_rna_test.csv')

dfATAC.iloc[trainInd].to_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/atac_atac_train.csv')
dfATAC.iloc[valInd].to_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/atac_atac_valid.csv')
dfATAC.iloc[test_index].to_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/atac_atac_test.csv')

