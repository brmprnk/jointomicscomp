import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.model_selection import StratifiedShuffleSplit

try:
    gefile = sys.argv[1]
    mefile = sys.argv[2]
    cnfile = sys.argv[3]
    tssfile = sys.argv[4]



except:
    gefile = 'data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena'
    mefile = 'data/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena'
    cnfile = 'data/TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes'
    tssfile = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/meta-data/codeTables/tissueSourceSite.tsv'

# gene expression
print('Starting GE\nLoading matrix...')
data = pd.read_table(gefile, index_col=0)

# selection based on std gave similar genes (jaccard > 0.9)
print('Filtering genes...')
gstd = data.mad(axis=1).sort_values(ascending=False)[:5000].index

data5000 = data.loc[gstd]

np.save('data/scaled-unscaled/unscaled.npy', np.array(data5000))

m = data5000.mean(axis=1)
s = data5000.std(axis=1)

data5000CS = data5000.apply(lambda x: x - m).apply(lambda x: x / s)

geneNamesGE = np.array(data5000CS.index)
sampleNamesGE = np.array(data5000CS.columns)

GE = np.array(data5000CS).T
GE[np.isnan(GE)] = 0.

np.save('data/scaled-unscaled/scaled.npy', GE)

sys.exit(0)
ll = np.zeros(GE.shape[1])

for j in range(GE.shape[1]):
    ll[j] = np.sum(norm.logpdf(GE[:, j]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(ll, bins=30)


# methylation
print('Starting ME\nReading big file...')
data = pd.read_table(mefile, index_col=0)

print('Converting to tss...')
with open('data/gene2probes.pkl', 'rb') as f:
    gene2probes = pickle.load(f)

X = np.zeros((data.shape[1], len(gene2probes)))
geneNamesME = []

includedProbes = set(data.index)

count = 0
for k in gene2probes:
    cols = gene2probes[k]
    cols2 = [c for c in cols if c in includedProbes]
    xx = np.array(data.loc[cols2])
    X[:, count] = np.nanmean(xx, axis=0)

    count += 1
    geneNamesME.append(k)

sampleNamesME = np.array(data.columns)

print('Filtering genes...')
data2 = pd.DataFrame(data=X.T, index=geneNamesME, columns=sampleNamesME)
gstd = data2.mad(axis=1).sort_values(ascending=False)[:5000].index

data5000 = data2.loc[gstd]

ME = np.array(data5000).T
sampleNamesME = np.array(data5000.columns)
geneNamesME = np.array(data5000.index)

ME[np.isnan(ME)] = 0.
eps = 1e-6
ME[ME == 0.] = eps
ME[ME == 1.] = 1 - eps



# copy numbers
print('Starting CN\nReading file...')
data = pd.read_table(cnfile, index_col=0)

print('Filtering genes...')
gstd = data.mad(axis=1).sort_values(ascending=False)[:5000].index

data5000 = data.loc[gstd]

geneNamesCN = np.array(data5000.index)
sampleNamesCN = np.array(data5000.columns)

CN = np.array(data5000).T
assert np.sum(np.isnan(CN)) == 0

CN -= np.min(CN)


print('Finding common samples...')
commonSamples = np.intersect1d(np.intersect1d(sampleNamesGE, sampleNamesME), sampleNamesCN)
sge = dict()
for i, s in enumerate(sampleNamesGE):
    sge[s] = i

sme = dict()
for i, s in enumerate(sampleNamesME):
    sme[s] = i

scn = dict()
for i, s in enumerate(sampleNamesCN):
    scn[s] = i

print('Making common dataset...')
cGE = np.zeros((commonSamples.shape[0], 5000))
cME = np.zeros((commonSamples.shape[0], 5000))
cCN = np.zeros((commonSamples.shape[0], 5000))

for i, s in enumerate(commonSamples):
    cGE[i] = GE[sge[s]]
    cME[i] = ME[sme[s]]
    cCN[i] = CN[scn[s]]


print('Getting cancer types...')
# cancer type
tss2c = dict()
with open(tssfile) as f:
	for line in f:
		tssCurrent, _, ct, _ = line.split('\t')
		assert tssCurrent not in tss2c
		tss2c[tssCurrent] = ct


name2code = {'Acute Myeloid Leukemia': 'LAML',
'Adrenocortical carcinoma': 'ACC',
'Bladder Urothelial Carcinoma': 'BLCA',
'Brain Lower Grade Glioma': 'LGG',
'Breast invasive carcinoma': 'BRCA',
'Cervical squamous cell carcinoma and endocervical adenocarcinoma': 'CESC',
'Cholangiocarcinoma': 'CHOL',
'Colon adenocarcinoma': 'COAD',
'Esophageal carcinoma ': 'ESCA',
'Glioblastoma multiforme': 'GBM',
'Head and Neck squamous cell carcinoma': 'HNSC',
'Kidney Chromophobe': 'KICH',
'Kidney renal clear cell carcinoma': 'KIRC',
'Kidney renal papillary cell carcinoma': 'KIRP',
'Liver hepatocellular carcinoma': 'LIHC',
'Lung adenocarcinoma': 'LUAD',
'Lung squamous cell carcinoma': 'LUSC',
'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma': 'DLBC',
'Mesothelioma': 'MESO',
'Ovarian serous cystadenocarcinoma': 'OV',
'Pancreatic adenocarcinoma': 'PAAD',
'Pheochromocytoma and Paraganglioma': 'PCPG',
'Prostate adenocarcinoma': 'PRAD',
'Rectum adenocarcinoma': 'READ',
'Sarcoma': 'SARC',
'Skin Cutaneous Melanoma': 'SKCM',
'Stomach adenocarcinoma': 'STAD',
'Testicular Germ Cell Tumors': 'TGCT',
'Thymoma': 'THYM',
'Thyroid carcinoma': 'THCA',
'Uterine Carcinosarcoma': 'UCS',
'Uterine Corpus Endometrial Carcinoma': 'UCEC',
'Uveal Melanoma': 'UVM',
}

name2number = {'Acute Myeloid Leukemia': 0,
'Adrenocortical carcinoma': 1,
'Bladder Urothelial Carcinoma': 2,
'Brain Lower Grade Glioma': 3,
'Breast invasive carcinoma': 4,
'Cervical squamous cell carcinoma and endocervical adenocarcinoma': 5,
'Cholangiocarcinoma': 6,
'Colon adenocarcinoma': 7,
'Esophageal carcinoma ': 8,
'Glioblastoma multiforme': 9,
'Head and Neck squamous cell carcinoma': 10,
'Kidney Chromophobe': 11,
'Kidney renal clear cell carcinoma': 12,
'Kidney renal papillary cell carcinoma': 13,
'Liver hepatocellular carcinoma': 14,
'Lung adenocarcinoma': 15,
'Lung squamous cell carcinoma': 16,
'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma': 17,
'Mesothelioma': 18,
'Ovarian serous cystadenocarcinoma': 19,
'Pancreatic adenocarcinoma': 20,
'Pheochromocytoma and Paraganglioma': 21,
'Prostate adenocarcinoma': 22,
'Rectum adenocarcinoma': 23,
'Sarcoma': 24,
'Skin Cutaneous Melanoma': 25,
'Stomach adenocarcinoma': 26,
'Testicular Germ Cell Tumors': 27,
'Thymoma': 28,
'Thyroid carcinoma': 29,
'Uterine Carcinosarcoma': 30,
'Uterine Corpus Endometrial Carcinoma': 31,
'Uveal Melanoma': 32}

classNames = []
for i, k in enumerate(name2number):
	assert name2number[k] == i
	classNames.append(name2code[k])

cancerType = np.array([name2number[tss2c[sam.split('-')[1]]] for sam in commonSamples])

print('Splitting train, validation, test...')
# First get test set (10%)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

sss1 = split1.split(commonSamples, cancerType)

# sss1 is a generator object, so use this weird syntax
trainValidInd = 0
testInd = 0
for i, j in sss1:
    trainValidInd = i
    testInd = j

# Let X denote data and let y denote labels
Xtest = commonSamples[testInd]

ytest = cancerType[testInd]

XtrainValid = commonSamples[trainValidInd]

ytrainValid = cancerType[trainValidInd]

# Now we get the validation set as 1/9 of the remaining 90%
split2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9)

sss2 = split2.split(XtrainValid, ytrainValid)

# sss2 is a generator object, so use this weird syntax
for i, j in sss2:
    trainInd = i
    validInd = j

finalTrainInd = trainValidInd[trainInd]
finalValidInd = trainValidInd[validInd]



print('Saving...')
np.save('data/GE.npy', cGE)
np.save('data/ME.npy', cME)
np.save('data/CN.npy', cCN)

np.save('data/genes_GE.npy', geneNamesGE)
np.save('data/genes_ME.npy', geneNamesME)
np.save('data/genes_CN.npy', geneNamesCN)

np.save('data/sampleNames.npy', commonSamples)
np.save('data/cancerType.npy', cancerType)

np.save('data/trainInd.npy', finalTrainInd)
np.save('data/validInd.npy', finalValidInd)
np.save('data/testInd.npy', testInd)
