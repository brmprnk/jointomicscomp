import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copy import deepcopy
import sys


which = 'PFI'

sampleNames = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/sampleNames.npy', allow_pickle=True)
trainInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/trainInd.npy')
validInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/validInd.npy')
testInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/testInd.npy')

cancerType = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerTypes.npy', allow_pickle=True)

model = 'uniport'
model2folder = {'uniport': 'UniPort'}
path = 'embeddings/results/test_%s_%s_%s/%s/embeddings.pkl'

annotations = pd.read_table('survival.deb', index_col=0)
annotatedSamples = np.array(annotations.index)

# keep required columns
annotations = annotations[['age_at_initial_pathologic_diagnosis', 'gender', 'PFI', 'PFI.time', 'OS', 'OS.time', 'cancer type abbreviation']]

# match the order of samples in metadata matrix and remove samples without embeddings
# also remove 22 samples with no metadata

tobedel = []
ind = []
for c, s in enumerate(sampleNames):
    ii = np.where(annotatedSamples == s)[0]
    if len(ii) == 1:
        ind.append(ii[0])

    else:
        assert len(ii) == 0
        tobedel.append(c)

sampleNames = np.delete(sampleNames, tobedel)

annotations = annotations.iloc[ind]

for s1, s2 in zip(sampleNames, list(annotations.index)):
    assert s1 == s2


dfType = pd.get_dummies(annotations['cancer type abbreviation'])
dfGender = pd.get_dummies(annotations['gender'])

surv = pd.concat((dfType, dfGender) , axis=1)
surv['PFI.time'] = annotations['PFI.time']
surv['PFI'] = annotations['PFI']
surv['age'] = annotations['age_at_initial_pathologic_diagnosis']


for dataset in ['GE_ME', 'GE_CNV']:
    ds = dataset.split('_')
    if ds[0] == 'CNV':
        ds[0] = 'CN'
    if ds[1] == 'CNV':
        ds[1] = 'CN'
    #


    with open(path % (dataset, model, dataset, model2folder[model]), 'rb') as f:
        embDict = pickle.load(f)


    zval = embDict['zvalidation']
    ztrain = embDict['ztrain']
    ztest = embDict['ztest']

    z = np.zeros((sampleNames.shape[0] + len(tobedel), zval.shape[1]))
    z[trainInd] = ztrain
    z[validInd] = zval
    z[testInd] = ztest

    z = np.delete(z, tobedel, axis=0)

    print('%s:\t%s\t%s' % (which, dataset, model))

    surv2 = pd.concat((surv, pd.DataFrame(z, index=annotations.index)), axis=1)

    surv2.iloc[:, 37:] = StandardScaler().fit_transform(np.array(surv2.iloc[:,37:]))

    surv3 = surv2.dropna(axis=0)
    strata = list(surv.columns)[:35]

    try:
        cpm = CoxPHFitter()
        cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
    except:
        try:
            cpm = CoxPHFitter(penalizer=0.1)
            cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
        except:
            try:
                cpm = CoxPHFitter(penalizer=1.0)
                cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
            except:
                print('Unable to fit Cox model. Too much colinearity')
                continue

    print('AIC: %.2f' % cpm.AIC_partial_)
    a = cpm.log_likelihood_ratio_test()
    print('LLR test p=%.5f' % a.p_value)
    # age is at 0
    pvalues_cox = cpm.summary['p'].iloc[1:]
    print('Significant factors: %d / %d' % (np.sum(pvalues_cox < 0.05), z.shape[1] ))
    print(np.where(pvalues_cox < 0.05)[0])
    print('\n\n', flush=True)
