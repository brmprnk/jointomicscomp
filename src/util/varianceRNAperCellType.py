import numpy as np
from scipy.stats import median_abs_deviation, variation


x = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD.npy')
labels = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD_cellType.npy')

labelNames = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD_cellTypes.npy')

testInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/testInd.npy')

xt = x[testInd]
yt = labels[testInd]

var = np.zeros(len(labelNames))
mad = np.zeros(var.shape)
cv = np.zeros(var.shape)

for i in range(var.shape[0]):
    data = xt[yt == i]

    var[i] = np.nanmedian(np.var(data, axis=0, ddof=1))
    mad[i] = np.median(median_abs_deviation(data, axis=0))
    cv[i] = np.nanmedian(variation(data, axis=0))

badCluster = {'HSPC', 'CD4 Proliferating', 'CD8 Proliferating', 'cDC2', 'CD14 Mono', 'CD16 Mono', 'ASDC', 'pDC', 'Doublet', 'NK Proliferating'}
badPerformance = []
goodPerformance = []

for i, k in enumerate(labelNames):
    if k in badCluster:
        badPerformance.append(cv[i])
    else:
        goodPerformance.append(cv[i])
