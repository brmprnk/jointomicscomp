import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copy import deepcopy
import sys


which = sys.argv[1]

sampleNames = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/sampleNames.npy')
trainInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/task1/trainInd.npy')
validInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/task1/validInd.npy')
testInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/task1/testInd.npy')

cancerType = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/cancerTypeStrings.npy')

model2folder = {'cgae': 'CGAE', 'mvib': 'MVIB', 'moe': 'MoE', 'poe': 'PoE'}
path = 'embeddings/results/test_%s_%s_%s/%s/embeddings.pkl'

annotations = pd.read_table('survival.deb', index_col=0)
annotatedSamples = np.array(annotations.index)

# keep required columns
annotations = annotations[['age_at_initial_pathologic_diagnosis', 'gender', 'PFI', 'PFI.time', 'OS', 'OS.time', 'cancer type abbreviation']]

# match the order of samples in metadata matrix and remove samples without embeddings
ind = []
for s in sampleNames:
    ii = np.where(annotatedSamples == s)[0]
    assert len(ii) == 1
    ind.append(ii[0])

annotations = annotations.iloc[ind]

dfType = pd.get_dummies(annotations['cancer type abbreviation'])
dfGender = pd.get_dummies(annotations['gender'])

surv = pd.concat((dfType, dfGender) , axis=1)
if which == 'PFI':
    surv['PFI.time'] = annotations['PFI.time']
    surv['PFI'] = annotations['PFI']
elif which == 'OS':
    surv['PFI.time'] = annotations['OS.time']
    surv['PFI'] = annotations['OS']


surv['age'] = annotations['age_at_initial_pathologic_diagnosis']
print('%s:\tonly age' % which)


surv2 = deepcopy(surv)

surv2.iloc[:, 37:] = StandardScaler().fit_transform(np.array(surv2.iloc[:,37:]))

surv3 = surv2.dropna(axis=0)
strata = list(surv.columns)[:35]

cpm = CoxPHFitter()
cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)

print('AIC: %.2f' % cpm.AIC_partial_)
a = cpm.log_likelihood_ratio_test()
print('LLR test p=%.5f' % a.p_value)
# age is at 0
print('\n\n')


for dataset in ['GE_ME', 'GE_CNV', 'ME_CNV']:
    ds = dataset.split('_')
    if ds[0] == 'CNV':
        ds[0] = 'GCN'
    if ds[1] == 'CNV':
        ds[1] = 'GCN'

    X1 = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/%s.npy' % ds[0])
    X2 = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/%s.npy' % ds[1])

    pca1 = PCA(n_components=32, whiten=False, svd_solver='full')
    X1pca = pca1.fit_transform(X1)

    pca2 = PCA(n_components=32, whiten=False, svd_solver='full')
    X2pca = pca2.fit_transform(X2)

    XX = np.hstack((X1pca, X2pca))
    print('%s:\t%s\tbaseline, PCs of 2 modalities' % (which, dataset))

    surv2 = pd.concat((surv, pd.DataFrame(XX, index=annotations.index)), axis=1)

    surv2.iloc[:, 37:] = StandardScaler().fit_transform(np.array(surv2.iloc[:,37:]))

    surv3 = surv2.dropna(axis=0)
    strata = list(surv.columns)[:35]

    cpm = CoxPHFitter()
    cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
    print('AIC: %.2f' % cpm.AIC_partial_)
    a = cpm.log_likelihood_ratio_test()
    print('LLR test p=%.5f' % a.p_value)
    # age is at 0
    pvalues_cox = cpm.summary['p'].iloc[1:]
    print('Significant factors: %d' % np.sum(pvalues_cox < 0.05))
    print(np.where(pvalues_cox < 0.05)[0])
    print('\n\n')

    print('%s:\t%s\tbaseline, modality 1' % (which, dataset))

    surv2 = pd.concat((surv, pd.DataFrame(X1, index=annotations.index)), axis=1)

    surv2.iloc[:, 37:] = StandardScaler().fit_transform(np.array(surv2.iloc[:,37:]))

    surv3 = surv2.dropna(axis=0)
    strata = list(surv.columns)[:35]

    cpm = CoxPHFitter()
    cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
    print('AIC: %.2f' % cpm.AIC_partial_)
    a = cpm.log_likelihood_ratio_test()
    print('LLR test p=%.5f' % a.p_value)
    # age is at 0
    pvalues_cox = cpm.summary['p'].iloc[1:]
    print('Significant factors: %d' % np.sum(pvalues_cox < 0.05))
    print(np.where(pvalues_cox < 0.05)[0])
    print('\n\n')

    print('%s:\t%s\tbaseline, modality 2' % (which, dataset))

    surv2 = pd.concat((surv, pd.DataFrame(X2, index=annotations.index)), axis=1)

    surv2.iloc[:, 37:] = StandardScaler().fit_transform(np.array(surv2.iloc[:,37:]))

    surv3 = surv2.dropna(axis=0)
    strata = list(surv.columns)[:35]

    cpm = CoxPHFitter()
    cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
    print('AIC: %.2f' % cpm.AIC_partial_)
    a = cpm.log_likelihood_ratio_test()
    print('LLR test p=%.5f' % a.p_value)
    # age is at 0
    pvalues_cox = cpm.summary['p'].iloc[1:]
    print('Significant factors: %d' % np.sum(pvalues_cox < 0.05))
    print(np.where(pvalues_cox < 0.05)[0])
    print('\n\n')


    print('%s:\t%s\tbaseline, PCA of modality 1' % (which, dataset))

    surv2 = pd.concat((surv, pd.DataFrame(X1pca, index=annotations.index)), axis=1)

    surv2.iloc[:, 37:] = StandardScaler().fit_transform(np.array(surv2.iloc[:,37:]))

    surv3 = surv2.dropna(axis=0)
    strata = list(surv.columns)[:35]

    cpm = CoxPHFitter()
    cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
    print('AIC: %.2f' % cpm.AIC_partial_)
    a = cpm.log_likelihood_ratio_test()
    print('LLR test p=%.5f' % a.p_value)
    # age is at 0
    pvalues_cox = cpm.summary['p'].iloc[1:]
    print('Significant factors: %d' % np.sum(pvalues_cox < 0.05))
    print(np.where(pvalues_cox < 0.05)[0])
    print('\n\n')

    print('%s:\t%s\tbaseline, pca of modality 2' % (which, dataset))

    surv2 = pd.concat((surv, pd.DataFrame(X2pca, index=annotations.index)), axis=1)

    surv2.iloc[:, 37:] = StandardScaler().fit_transform(np.array(surv2.iloc[:,37:]))

    surv3 = surv2.dropna(axis=0)
    strata = list(surv.columns)[:35]

    cpm = CoxPHFitter()
    cpm.fit(surv3,  duration_col='PFI.time', event_col='PFI', strata=strata)
    print('AIC: %.2f' % cpm.AIC_partial_)
    a = cpm.log_likelihood_ratio_test()
    print('LLR test p=%.5f' % a.p_value)
    # age is at 0
    pvalues_cox = cpm.summary['p'].iloc[1:]
    print('Significant factors: %d' % np.sum(pvalues_cox < 0.05))
    print(np.where(pvalues_cox < 0.05)[0])
    print('\n\n')




    # for model in ['cgae', 'moe', 'poe']:
    for model in ['cgae', 'mvib', 'moe', 'poe']:

        with open(path % (dataset, model, dataset, model2folder[model]), 'rb') as f:
            embDict = pickle.load(f)

        if len(embDict.keys()) == 3:
            zval = embDict['zvalidation']
            ztrain = embDict['ztrain']
            ztest = embDict['ztest']

            z = np.zeros((sampleNames.shape[0], zval.shape[1]))
            z[trainInd] = ztrain
            z[validInd] = zval
            z[testInd] = ztest

            llz = [z]

        else:
            z1val = embDict['z1validation']
            z1train = embDict['z1train']
            z1test = embDict['z1test']

            z2val = embDict['z2validation']
            z2train = embDict['z2train']
            z2test = embDict['z2test']

            z1 = np.zeros((sampleNames.shape[0], z1val.shape[1]))
            z1[trainInd] = z1train
            z1[validInd] = z1val
            z1[testInd] = z1test

            z2 = np.zeros((sampleNames.shape[0], z2val.shape[1]))
            z2[trainInd] = z2train
            z2[validInd] = z2val
            z2[testInd] = z2test


            llz = [z1, z2]

        #pvalues = np.zeros((len(llz), llz[0].shape[1]))
        for i, z in enumerate(llz):
            print('%s:\t%s\t%s\t%d' % (which, dataset, model, i))

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
            print('\n' )
