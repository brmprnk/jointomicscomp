import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats.mstats import friedmanchisquare
from scipy.stats import wilcoxon
from scipy.special import binom
from scipy.cluster.hierarchy import linkage
import sys

def customBarPlot(width, ll1, ll2, err1, err2, ax, modalities, models, ylabel='Log likelihood', ylim=(-2000, 2000)):

    x = np.arange(len(ll1))  # the label locations

    #ll1 = ll1 - ll1[0]
    #ll2 = ll2 - ll2[0]

    rects2 = ax.bar(x + width/2, ll2, width=width, yerr=err2, color='C1',label=modalities[1], edgecolor='k', error_kw=dict(lw=1, capsize=3, capthick=1))
    rects1 = ax.bar(x - width/2, ll1, width=width, yerr=err1, color='C0',label=modalities[0], edgecolor='k', error_kw=dict(lw=1, capsize=3, capthick=1))
    ax.axhline(ll1[0], color='C0', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(ll2[0], color='C1', linestyle='--', linewidth=1, alpha=0.5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel, fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    ax.set_ylim(ylim)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    return ax


def twoBarPlot2(width, ll1, ll2, err1, err2, fig, modalities, models):

    x = np.arange(len(ll1))  # the label locations

    #ll1 = ll1 - ll1[0]
    #ll2 = ll2 - ll2[0]
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    rects2 = ax2.bar(x, ll2, width=width, yerr=err2, label=modalities[1], ecolor='k')
    rects1 = ax1.bar(x, ll1, width=width, yerr=err1, label=modalities[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Log likelihood')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()

    ax2.set_ylabel('Log likelihood')

    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    return [ax1, ax2]


plt.close('all')

resultDict = dict()
resultDict['GE_ME'] = {
'GLM': 'embeddings/results/test-baseline_GE_ME/baseline/test-baselinetest_performance_per_datapoint.pkl',
'MOFA+': 'src/MOFA2/imputation_GEME_performancePerDatapont.pkl',
'MCIA': 'src/MCIA/imputation_GEME_performancePerDatapont.pkl',
'CGAE': 'embeddings/results/test_GE_ME_cgae_GE_ME/CGAE/test_performance_per_datapoint.pkl',
'CVAE': 'embeddings/results/test_GE_ME_cvae_GE_ME/CVAE/test_performance_per_datapoint.pkl',
'MoE': 'embeddings/results/test_GE_ME_moe_GE_ME/MoE/test_performance_per_datapoint.pkl',
'PoE': 'embeddings/results/test_GE_ME_poe_GE_ME/PoE/test_performance_per_datapoint.pkl',
'uniport': 'embeddings/results/test_GE_ME_uniport_GE_ME/UniPort/test_performance_per_datapoint.pkl',
'labelnames': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerTypes.npy',
'labels': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerType.npy',
'test_ind': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/testInd.npy'}


resultDict['GE_CNV'] = {
'GLM': 'embeddings/results/likelihood-tcga-baseline-GECNV_GE_CNV/baseline/likelihood-tcga-baseline-GECNVtest_performance_per_datapoint.pkl',
'MOFA+': 'src/MOFA2/imputation_GECNV_performancePerDatapont.pkl',
'MCIA': 'src/MCIA/imputation_GECNV_performancePerDatapont.pkl',
'CGAE': 'embeddings/results/test_GE_CNV_cgae_GE_CNV/CGAE/test_performance_per_datapoint.pkl',
'CVAE': 'embeddings/results/test_GE_CNV_cvae_GE_CNV/CVAE/test_performance_per_datapoint.pkl',
'MoE': 'embeddings/results/test_GE_CNV_moe_GE_CNV/MoE/test_performance_per_datapoint.pkl',
'PoE': 'embeddings/results/test_GE_CNV_poe_GE_CNV/PoE/test_performance_per_datapoint.pkl',
'uniport': 'embeddings/results/test_GE_CNV_uniport_GE_CNV/UniPort/test_performance_per_datapoint.pkl',
'labelnames': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerTypes.npy',
'labels': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerType.npy',
'test_ind': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/testInd.npy'}

resultDict['RNA_ADT'] = {
'GLM': 'results/tmpnbcite/results/test-cite-baseline_RNA_ADT/baseline/test-cite-baselinetest_performance_per_datapoint.pkl',
'MOFA+': 'src/MOFA2/imputation_RNAADT_performancePerDatapont.pkl',
'MCIA': '',
'CGAE': 'embeddings/results/test_RNA_ADT_cgae_RNA_ADT/CGAE/test_performance_per_datapoint.pkl',
'CVAE': 'embeddings/results/test_RNA_ADT_cvae_RNA_ADT/CVAE/test_performance_per_datapoint.pkl',
'MoE': 'embeddings/results/test_RNA_ADT_moe_RNA_ADT/MoE/test_performance_per_datapoint.pkl',
'PoE': 'embeddings/results/test_RNA_ADT_poe_RNA_ADT/PoE/test_performance_per_datapoint.pkl',
'totalvi': 'embeddings/results/test_RNA_ADT_totalvi_RNA_ADT/totalVI/test_performance_per_datapoint.pkl',
'uniport': 'embeddings/results/test_RNA_ADT_uniport_RNA_ADT/UniPort/test_performance_per_datapoint.pkl',
'labelnames': 'data/scvi-cite/celltypes.npy',
'labels': 'data/scvi-cite/celltype.npy',
'test_ind': 'data/scvi-cite/testInd.npy'}

resultDict['RNA_ATAC'] = {
'GLM': 'embeddings/results/test_RNA_ATAC_baseline_RNA_ATAC/baseline/test_RNA_ATAC_baselinetest_performance_per_datapoint.pkl',
'MOFA+': 'src/MOFA2/imputation_RNAATAC_performancePerDatapont.pkl',
'MCIA': 'src/MCIA/imputation_RNAATAC_performancePerDatapont.pkl',
'CGAE': 'embeddings/results/test_RNA_ATAC_cgae_RNA_ATAC/CGAE/test_performance_per_datapoint.pkl',
'CVAE': 'embeddings/results/test_RNA_ATAC_cvae_RNA_ATAC/CVAE/test_performance_per_datapoint.pkl',
'MoE': 'embeddings/results/test_RNA_ATAC_moe_RNA_ATAC/MoE/test_performance_per_datapoint.pkl',
'PoE': 'embeddings/results/test_RNA_ATAC_poe_RNA_ATAC/PoE/test_performance_per_datapoint.pkl',
'uniport': 'embeddings/results/test_RNA_ATAC_uniport_RNA_ATAC/UniPort/test_performance_per_datapoint.pkl',
'labelnames': 'data/scatac/celltypes_l2.npy',
'labels': 'data/scatac/celltype_l2.npy',
'test_ind': 'data/scatac/testInd.npy'}

w = 0.4

datasets = ['GE_ME', 'GE_CNV', 'RNA_ADT', 'RNA_ATAC']
#datasets = ['GE_ME', 'GE_CNV', 'RNA_ATAC']


myylims = [(-12300, 10500), (-32000, 3000), (-5500, 0), (-6000, 0)]
myvmins = [[-12308.2295, 0], [-20000, 15000], [-5855., -5898.], [-4300, -32190.49]]



for i, ds in enumerate(datasets):
    modalities = ds.split('_')


    labels = ['%s from %s' % (modalities[0], modalities[1]), '%s from %s' % (modalities[1], modalities[0])]

    obs1 = []
    obs2 = []

    if i == 2:
        models = ['GLM', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE', 'totalvi', 'uniport']
        modelNames = ['GLM', 'MOFA+', 'CGVAE', 'ccVAE', 'PoE', 'MoE', 'totalVI', 'UniPort']

    else:
        models = ['GLM', 'MCIA', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE', 'uniport']
        modelNames = ['GLM', 'MCIA', 'MOFA+', 'CGVAE', 'ccVAE', 'PoE', 'MoE', 'UniPort']

    fwerCorr = binom(len(models), 2)

    modelsImputeBoth = [m for m in models if m not in set(['uniport', 'totalvi'])]
    modelsNamesImputeBoth = [m for m in modelNames if m not in set(['UniPort', 'totalVI'])]

    #
    # print(models)
    # print(modelsImputeBoth)
    # continue


    ll1 = np.zeros(len(models))
    ll2 = np.zeros(len(models))

    std1 = np.zeros(len(models))
    std2 = np.zeros(len(models))
    fwerCorr = binom(len(models), 2)

    labelsAll = np.load(resultDict[ds]['labels'])
    labelNames = np.load(resultDict[ds]['labelnames'], allow_pickle=True)
    tstInd = np.load(resultDict[ds]['test_ind'])

    labels = labelsAll[tstInd]
    labels = list(pd.Series(labels).map(pd.Series(labelNames).to_dict()))


    for j, model in enumerate(models):
        with open(resultDict[ds][model], 'rb') as f:
             dd = pickle.load(f)

        if 'LL1/2' in dd.keys():

            key12 = 'LL1/2'
            key21 = 'LL2/1'

            # n = dd[key12].shape[0]

            numbers = dd[key12].numpy()
            if model == 'PoE':
                numbers *= -1
                # by mistake in per datapoint eval of poe the loss is returned

            obs1.append(numbers)
            ll1[j] = np.mean(numbers)
            std1[j] = np.std(numbers, ddof=1)

            numbers = dd[key21].numpy()
            if model == 'PoE':
                numbers *= -1
                # by mistake in per datapoint eval of poe the loss is returned

            obs2.append(numbers)
            ll2[j] = np.mean(numbers)
            std2[j] = np.std(numbers, ddof=1)


        elif 'loss' not in dd and '1from2' in dd:

            key12 = '1from2'
            key21 = '2from1'



            obs1.append(-1 * dd[key12]['loss'].numpy())
            ll1[j] = np.mean(-1 * dd[key12]['loss'].numpy())
            std1[j] = np.std(-1 * dd[key12]['loss'].numpy(), ddof=1)

            obs2.append(-1 * dd[key21]['loss'].numpy())
            ll2[j] = np.mean(-1 * dd[key21]['loss'].numpy())
            std2[j] = np.std(-1 * dd[key21]['loss'].numpy(), ddof=1)

        else:
            if model == 'uniport':
                obs1.append(np.nan * np.ones(dd['loss'].shape[0]))
                ll1[j] = np.nan
                std1[j] = np.nan

                obs2.append(-1 * dd['loss'].numpy())
                ll2[j] = np.mean(-1 * dd['loss'].numpy())
                std2[j] = np.std(-1 * dd['loss'].numpy(), ddof=1)
            else:
                assert model == 'totalvi'
                obs1.append(np.nan * np.ones(dd['LL2/1'].shape[0]))
                ll1[j] = np.nan
                std1[j] = np.nan

                obs2.append(dd['LL2/1'].cpu().detach().numpy())
                ll2[j] = np.mean(dd['LL2/1'].cpu().detach().numpy())
                std2[j] = np.std(dd['LL2/1'].cpu().detach().numpy(), ddof=1)


    rnaImp = np.array(obs1).flatten()
    adtImp = np.array(obs2).flatten()
    mmm = []
    for m in modelNames:
        mmm += [m] * len(obs1[0])

    mmm = mmm + mmm
    label = labels * (2 * len(models))

    modality = [modalities[0] + ' from ' + modalities[1]] * len(rnaImp)
    modality += [modalities[1] + ' from ' + modalities[0]] * len(adtImp)

    hugeDF = pd.DataFrame({'Log likelihood': np.hstack((rnaImp, adtImp)), 'Model': mmm, 'modality': modality, 'label': label})

    print('ANOVA, rec of modality 1')
    stat, p = friedmanchisquare(obs1)
    print('p = %.20f' % p)
    #
    #
    print('ANOVA, rec of modality 2')
    stat, p = friedmanchisquare(obs2)
    #
    print('p = %.20f' % p)
    #
    print('Post hoc tests')
    for kk in range(len(models) - 1):
        for ll in range(kk + 1, len(models)):
            if models[kk] not in set(['uniport', 'totalvi']) and models[ll] not in set(['uniport', 'totalvi']):
                pval = wilcoxon(obs1[kk], obs1[ll], correction=True)[1] * fwerCorr
                print('Modality 1, %s vs %s: p = %.20f' % (models[kk], models[ll], pval))
   
                pval = wilcoxon(obs2[kk], obs2[ll], correction=True)[1] * fwerCorr
                print('Modality 2, %s vs %s: p = %.20f' % (models[kk], models[ll], pval))


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # ax = customBarPlot(w, ll1, ll2, std1, std2, ax, labels, models, ylim=myylims[i])


    fig = plt.figure()
    ax = fig.add_subplot(111)


    sns.boxplot(x='Model', y='Log likelihood', hue='modality', data=hugeDF, ax=ax)
    means = np.array(hugeDF[hugeDF['Model'] == 'GLM'].groupby('modality').median()['Log likelihood']).reshape(2,)

    if i == 0:
        ax.axhline(means[0], color='C0', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(means[1], color='C1', linestyle='--', linewidth=1, alpha=0.5)
    else:
        ax.axhline(means[1], color='C0', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(means[0], color='C1', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylim(myylims[i])
    # plt.grid()
    ax.tick_params(axis='y',labelsize=8)
    fig.savefig('figures/rebuttal/imputation%s_%s.eps' % (modalities[0], modalities[1]), dpi=1200)
    fig.savefig('figures/rebuttal/imputation%s_%s.png' % (modalities[0], modalities[1]), dpi=600)

    hugeDF.dropna(axis=0, inplace=True)

    if i > 1:
        medianModality1PerClass = hugeDF[hugeDF['modality'] == modality[0]].groupby(['Model', 'label']).median().values.reshape(len(modelsImputeBoth), np.unique(labelNames).shape[0])
        medianModality2PerClass = hugeDF[hugeDF['modality'] == modality[-1]].groupby(['Model', 'label']).median().values.reshape(len(models), np.unique(labelNames).shape[0])

        print(np.min(medianModality1PerClass), np.max(medianModality1PerClass))
        print(np.min(medianModality2PerClass), np.max(medianModality2PerClass))

        zmodel = linkage(medianModality1PerClass, method='average', metric='correlation')
        zctype = linkage(medianModality1PerClass.T, method='average', metric='euclidean')

        ff = sns.clustermap(pd.DataFrame(medianModality1PerClass, index=sorted(modelsNamesImputeBoth), columns=sorted(labelNames)), cmap='Greens', row_linkage=zmodel, col_linkage=zctype, vmin=myvmins[i][0])
        plt.tight_layout()
        ff.savefig('figures/rebuttal/imputationPerClass_%s_%s.eps' % (modalities[0], modalities[1]), dpi=1200)
        ff.savefig('figures/rebuttal/imputationPerClass_%s_%s.svg' % (modalities[0], modalities[1]), dpi=1200)
        ff.savefig('figures/rebuttal/imputationPerClass_%s_%s.png' % (modalities[0], modalities[1]), dpi=600)

        zmodel = linkage(medianModality2PerClass, method='average', metric='correlation')
        zctype = linkage(medianModality2PerClass.T, method='average', metric='euclidean')

        ff = sns.clustermap(pd.DataFrame(medianModality2PerClass, index=sorted(modelNames), columns=sorted(labelNames)), cmap='Greens', row_linkage=zmodel, col_linkage=zctype, vmin=myvmins[i][0])
        plt.tight_layout()
        ff.savefig('figures/rebuttal/imputationPerClass_%s_%s.eps' % (modalities[1], modalities[0]), dpi=1200)
        ff.savefig('figures/rebuttal/imputationPerClass_%s_%s.svg' % (modalities[1], modalities[0]), dpi=1200)
        ff.savefig('figures/rebuttal/imputationPerClass_%s_%s.png' % (modalities[1], modalities[0]), dpi=600)
        sys.exit(0)

# to save memory
plt.close('all')
resultDictClf = dict()
resultDictClf['RNA_ADT'] = {
######################################################################################################################################
'PCA': 'results/tmpnbcite/results/test-cite-baseline_RNA_ADT/baseline/test-cite-baselineresults_pickle',
######################################################################################################################################
'MOFA+': 'src/MOFA2/MOFA_task2_results.pkl',
'CGAE': 'embeddings/results/test_RNA_ADT_cgae_RNA_ADT/CGAE/CGAE_task2_results.pkl',
'CVAE': 'embeddings/results/test_RNA_ADT_cvae_RNA_ADT/CVAE/CVAE_task2_results.pkl',
'MoE': 'embeddings/results/test_RNA_ADT_moe_RNA_ADT/MoE/CGAE_task2_results.pkl',
'PoE': 'embeddings/results/test_RNA_ADT_poe_RNA_ADT/PoE/PoE_task2_results.pkl',
'totalvi': 'embeddings/results/test_RNA_ADT_totalvi_RNA_ADT/totalVI/totalvi_task2_results.pkl',
'uniport' : 'embeddings/results/test_RNA_ADT_uniport_RNA_ADT/UniPort/uniport_task2_results.pkl'}

labelNames = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltypes_l3.npy', allow_pickle=True)
labelNames = np.delete(labelNames, 19)
models = ['PCA', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE', 'totalvi', 'uniport']
modelNames = ['PCA', 'MOFA+', 'CGVAE', 'ccVAE', 'PoE', 'MoE', 'totalVI', 'UniPort']

ds = 'RNA_ADT'


modalities = ds.split('_')
mcc = np.zeros((2,len(models)))
f1 = np.zeros((2, len(models), len(labelNames)))

mccCI = np.zeros((2, len(models), 2))
#f1CI = np.zeros((len(models), 57,2))

freq = np.load('l3_class_frequencies_test_data.npy')



omicNames = ['RNA+ADT', 'RNA', 'ADT']

names = []
svmf1 = []
mlpf1 = []
classes = []

figTripleBar = plt.figure()
axTripleBar = figTripleBar.add_subplot(111)

figTripleBarSVM = plt.figure()
axTripleBarSVM = figTripleBarSVM.add_subplot(111)

ccTripleBar = ['C3', 'C1', 'C2']

wTripleBar = 0.2

offestsTripleBar = [-0.5, 1.5, 0.5]

for oo, omic in enumerate(['omic1+2', 'omic1', 'omic2']):

    # if omic == 'omic1':
    #     # RNA, includes uniport
    #     models = ['PCA', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE', 'totalvi', 'uniport']
    #     modelNames = ['PCA', 'MOFA+', 'CGVAE', 'ccVAE', 'PoE', 'MoE', 'totalVI', 'UniPort']
    # else:
    #     models = ['PCA', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE', 'totalvi']
    #     modelNames = ['PCA', 'MOFA+', 'CGVAE', 'ccVAE', 'PoE', 'MoE', 'totalVI']

    for j, model in enumerate(models):
        with open(resultDictClf[ds][model], 'rb') as f:
             dd = pickle.load(f)

        print(model)
        for k, classifier in enumerate(['', 'mlp_']):
            try:
                mcc[k,j] = dd[classifier + omic]['mcc']
                mccCI[k,j] = dd[classifier + omic]['CIs'][:,-1]

                mccCI[k, j, 0] = mcc[k,j] - mccCI[k,j,0]
                mccCI[k, j, 1] = mccCI[k,j,1] - mcc[k,j]


                tmp = dd[classifier + omic]['f1']
            except KeyError:
                try:
                    mcc[k,j] = dd[omic + '-mlp']['mcc']
                    mccCI[k,j] = dd[omic + '-mlp']['CIs'][:,-1]

                    mccCI[k, j, 0] = mcc[k, j] - mccCI[k, j,0]
                    mccCI[k, j, 1] = mccCI[k, j,1] - mcc[k, j]

                    tmp = dd[omic + '-mlp']['f1']
                except KeyError:
                    assert model == 'uniport'
                    mcc[k,j] = np.nan
                    mccCI[k,j] = np.nan


            if tmp.shape[0] == len(labelNames):
                f1[k,j] = tmp
            else:
                # this model made predictions for tye class not in the test set
                f1[k,j] = np.delete(tmp, 19)

    for jjj in range(f1.shape[1]):
        for clsi in range(f1.shape[2]):
            names.append(modelNames[jjj] + '_' + omicNames[oo])
            svmf1.append(f1[0,jjj, clsi])
            mlpf1.append(f1[1,jjj, clsi])
            classes.append(labelNames[clsi])


    print(omic)
    print('SVM: all failed')
    print(labelNames[np.max(f1[0], 0) < freq])

    print('MLP: all failed')
    print(labelNames[np.max(f1[1],0) < freq])

    print('SVM: PCA failed')
    print(labelNames[f1[0,0] < freq])

    print('MLP: PCA failed')
    print(labelNames[f1[1,0] < freq])

    with open(resultDictClf[ds]['PCA'], 'rb') as f:
         dd = pickle.load(f)

    confmatPCA = dd[omic]['confMat']
    if confmatPCA.shape != (len(labelNames), len(labelNames)):
        print('DANGER!')


    pcaFails = np.where(f1[1,0] < freq)[0]
    for clsind in pcaFails:
        print('Class: %s\tfrequency: %.6f' % (labelNames[clsind], freq[clsind]))

        for l, mm in enumerate(models):
            print('%s:\t%.6f' % (mm, f1[1,l,clsind]))


        print(labelNames[confmatPCA[clsind] > 0])
    #confmatPCA = np.delete(confmat, 19, axis=0)
    #print(conf)
    #conf

    axTripleBar.bar(np.arange(len(models)) - offestsTripleBar[oo]*wTripleBar, mcc[1], width=wTripleBar, yerr=mccCI[1].T, color=ccTripleBar[oo], label=omicNames[oo], align='edge', error_kw=dict(lw=1, capsize=3, capthick=1))

    axTripleBarSVM.bar(np.arange(len(models)) - offestsTripleBar[oo]*wTripleBar, mcc[0], width=wTripleBar, yerr=mccCI[0].T, color=ccTripleBar[oo], label=omicNames[oo], align='edge', error_kw=dict(lw=1, capsize=3, capthick=1))

    if oo == 0:
        axTripleBar.axhline(mcc[1,0], color='k', linestyle='--')
        axTripleBarSVM.axhline(mcc[0,0], color='k', linestyle='--')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if omic == 'omic1':
        ax = customBarPlot(w, mcc[0], mcc[1], mccCI[0].T, mccCI[1].T, ax, ['SVM', 'MLP'], modelNames, 'MCC', (0,1))
    else:
        ax = customBarPlot(w, mcc[0][:-1], mcc[1][:-1], mccCI[0][:-1].T, mccCI[1][:-1].T, ax, ['SVM', 'MLP'], modelNames[:-1], 'MCC', (0,1))
    ax.legend(loc='lower left')
    ax.set_title(omicNames[oo])

    fig.savefig('figures/rebuttal/classification_l3_mcc_%s.eps' % omic, dpi=1200)
    fig.savefig('figures/rebuttal/classification_l3_mcc_%s.png' % omic, dpi=600)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.scatter(f1[0,0], f1[0,1], color='C0', label='SVM')
    ax.scatter(f1[1,0], f1[1,1], color='C1', label='MLP')
    gmd = np.linspace(0,1,10)
    ax.plot(gmd, gmd, 'k--')
    ax.set_xlabel('F1 score PCA')
    ax.set_ylabel('F1 score MOFA+')
    ax.legend()

    ax = fig.add_subplot(1,2,2)
    ax.scatter(f1[0,0], f1[0,4], color='C0', label='SVM')
    ax.scatter(f1[1,0], f1[1,4], color='C1', label='MLP')

    ax.plot(gmd, gmd, 'k--')
    ax.set_xlabel('F1 score PCA')
    ax.set_ylabel('F1 score PoE')
    plt.suptitle(omicNames[oo])
    plt.tight_layout()

    plt.close(fig)
    fig.savefig('figures/rebuttal/classification_l3_f1_%s.eps' % omic, dpi=1200)
    fig.savefig('figures/rebuttal/classification_l3_f1_%s.png' % omic, dpi=600)

axTripleBar.set_xticks(np.arange(len(modelNames)))
axTripleBar.set_xticklabels(modelNames)
axTripleBar.legend(loc='lower left')
axTripleBar.set_ylabel('MCC', fontsize=14)

axTripleBarSVM.set_xticks(np.arange(len(modelNames)))
axTripleBarSVM.set_xticklabels(modelNames)
axTripleBarSVM.legend(loc='lower left')
axTripleBarSVM.set_ylabel('MCC', fontsize=14)

figTripleBar.savefig('figures/rebuttal/mcc_tripleBar_mlp.png', dpi=600)
figTripleBar.savefig('figures/rebuttal/mcc_tripleBar_mlp.eps', dpi=1200)

figTripleBarSVM.savefig('figures/rebuttal/mcc_tripleBar_svm.png', dpi=600)
figTripleBarSVM.savefig('figures/rebuttal/mcc_tripleBar_svm.eps', dpi=1200)

plt.close('all')


df = pd.DataFrame({'model': names, 'cell type': classes, 'SVM F1': svmf1, 'MLP F1': mlpf1})

nnn = []
svmf1 = np.zeros((len(omicNames) * len(models), len(labelNames) ))
mlpf1 = np.zeros(svmf1.shape)


for oo, omic in enumerate(['omic1+2', 'omic1', 'omic2']):
    for j, model in enumerate(models):
        with open(resultDictClf[ds][model], 'rb') as f:
             dd = pickle.load(f)

        print(model)
        if model == 'uniport' and omic != 'omic1':
            continue

        for k, classifier in enumerate(['', 'mlp_']):
            try:
                mcc[k,j] = dd[classifier + omic]['mcc']
                mccCI[k,j] = dd[classifier + omic]['CIs'][:,-1]

                mccCI[k, j, 0] = mcc[k,j] - mccCI[k,j,0]
                mccCI[k, j, 1] = mccCI[k,j,1] - mcc[k,j]


                tmp = dd[classifier + omic]['f1']
            except KeyError:
                mcc[k,j] = dd[omic + '-mlp']['mcc']
                mccCI[k,j] = dd[omic + '-mlp']['CIs'][:,-1]

                mccCI[k, j, 0] = mcc[k, j] - mccCI[k, j,0]
                mccCI[k, j, 1] = mccCI[k, j,1] - mcc[k, j]

                tmp = dd[omic + '-mlp']['f1']


            if tmp.shape[0] == 56:
                f1[k,j] = tmp
            else:
                # this model made predictions for tye class not in the test set
                assert tmp.shape[0] == 57
                f1[k,j] = np.delete(tmp, 19)

            if k == 0:
                svmf1[len(omicNames)*j + oo] = f1[k,j]
            else:
                mlpf1[len(omicNames)*j + oo] = f1[k,j]
            nnn.append(model + '_' + omicNames[oo])


svmf1 = np.delete(svmf1, [-1,-3], axis=0)
mlpf1 = np.delete(mlpf1, [-1,-3], axis=0)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

sns.heatmap(svmf1.T, square=True, vmin=0, vmax=1, cbar=True, ax=ax, yticklabels=labelNames)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.set_xticks([3*i+1.5 for i in range(len(modelNames)-1)] + [21.5])
ax.set_xticklabels(modelNames)

ax.tick_params(axis='x', which='major', rotation=45, labelsize=9)

for i in range(1,len(modelNames)):
    ax.axvline(3 * i, color='b')

ax.set_title('F1 score per class, SVM')
fig.savefig('figures/rebuttal/svm_performance_per_class.eps', dpi=600)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

sns.heatmap(mlpf1.T, square=True, vmin=0, vmax=1, cbar=True, ax=ax, yticklabels=labelNames)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.set_xticks([3*i+1.5 for i in range(len(modelNames)-1)] + [21.5])
ax.set_xticklabels(modelNames)

ax.tick_params(axis='x', which='major', rotation=45, labelsize=9)

for i in range(1,len(modelNames)):
    ax.axvline(3 * i, color='b')

#plt.setp(ax.get_yticklabels(), {'fontsize': 10})

ax.set_title('F1 score per class, MLP')


fig.savefig('figures/rebuttal/mlp_performance_per_class.eps', dpi=600)
plt.close('all')

sys.exit(0)
################################################################################
# ATAC
resultDictClf['RNA_ATAC'] = {
######################################################################################################################################
'PCA': 'embeddings/results/test_RNA_ATAC_baseline_RNA_ATAC/baseline/test_RNA_ATAC_baselineresults_pickle',
######################################################################################################################################
'MOFA+': 'src/MOFA2/atac_MOFA_task2_results.pkl',
'MCIA': 'src/MCIA/atac_MCIA_task2_results.pkl',
'CGAE': 'embeddings/results/test_RNA_ATAC_cgae_RNA_ATAC/CGAE/CGAE_task2_results.pkl',
'CVAE': 'embeddings/results/test_RNA_ATAC_cvae_RNA_ATAC/CVAE/CVAE_task2_results.pkl',
'MoE': 'embeddings/results/test_RNA_ATAC_moe_RNA_ATAC/MoE/CGAE_task2_results.pkl',
'PoE': 'embeddings/results/test_RNA_ATAC_poe_RNA_ATAC/PoE/PoE_task2_results.pkl',
'uniport' : 'embeddings/results/test_RNA_ATAC_uniport_RNA_ATAC/UniPort/uniport_task2_results.pkl'}

labelNames = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/celltypes_l2.npy', allow_pickle=True)
models = ['PCA', 'MCIA','MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE', 'uniport']
modelNames = ['PCA', 'MCIA', 'MOFA+', 'CGVAE', 'ccVAE', 'PoE', 'MoE', 'UniPort']

ds = 'RNA_ATAC'


modalities = ds.split('_')
mcc = np.zeros((2,len(models)))
f1 = np.zeros((2, len(models), len(labelNames)))

mccCI = np.zeros((2, len(models), 2))
#f1CI = np.zeros((len(models), 57,2))

# freq = np.load('l3_class_frequencies_test_data.npy')

omicNames = ['RNA+ATAC', 'RNA', 'ATAC']

names = []
svmf1 = []
mlpf1 = []
classes = []

figTripleBar = plt.figure()
axTripleBar = figTripleBar.add_subplot(111)

figTripleBarSVM = plt.figure()
axTripleBarSVM = figTripleBarSVM.add_subplot(111)

ccTripleBar = ['C3', 'C1', 'C2']

wTripleBar = 0.2

offestsTripleBar = [-0.5, 1.5, 0.5]

for oo, omic in enumerate(['omic1+2', 'omic1', 'omic2']):

    for j, model in enumerate(models):
        with open(resultDictClf[ds][model], 'rb') as f:
             dd = pickle.load(f)

        print(model)
        for k, classifier in enumerate(['', 'mlp_']):
            try:
                mcc[k,j] = dd[classifier + omic]['mcc']
                mccCI[k,j] = dd[classifier + omic]['CIs'][:,-1]

                mccCI[k, j, 0] = mcc[k,j] - mccCI[k,j,0]
                mccCI[k, j, 1] = mccCI[k,j,1] - mcc[k,j]


                tmp = dd[classifier + omic]['f1']
            except KeyError:
                try:
                    mcc[k,j] = dd[omic + '-mlp']['mcc']
                    mccCI[k,j] = dd[omic + '-mlp']['CIs'][:,-1]

                    mccCI[k, j, 0] = mcc[k, j] - mccCI[k, j,0]
                    mccCI[k, j, 1] = mccCI[k, j,1] - mcc[k, j]

                    tmp = dd[omic + '-mlp']['f1']
                except KeyError:
                    assert model == 'uniport'
                    mcc[k,j] = np.nan
                    mccCI[k,j] = np.nan


            if tmp.shape[0] == len(labelNames):
                f1[k,j] = tmp
            else:
                # this model made predictions for tye class not in the test set
                print('!!!!')
                f1[k,j] = np.delete(tmp, 19)

    for jjj in range(f1.shape[1]):
        for clsi in range(f1.shape[2]):
            names.append(modelNames[jjj] + '_' + omicNames[oo])
            svmf1.append(f1[0,jjj, clsi])
            mlpf1.append(f1[1,jjj, clsi])
            classes.append(labelNames[clsi])


    print(omic)
    print('SVM: all failed')
    # print(labelNames[np.max(f1[0], 0) < freq])

    print('MLP: all failed')
    # print(labelNames[np.max(f1[1],0) < freq])

    print('SVM: PCA failed')
    # print(labelNames[f1[0,0] < freq])

    print('MLP: PCA failed')
    # print(labelNames[f1[1,0] < freq])

    with open(resultDictClf[ds]['PCA'], 'rb') as f:
         dd = pickle.load(f)

    confmatPCA = dd[omic]['confMat']
    if confmatPCA.shape != (len(labelNames), len(labelNames)):
        print('DANGER!')


    # pcaFails = np.where(f1[1,0] < freq)[0]
    # for clsind in pcaFails:
    #     print('Class: %s\tfrequency: %.6f' % (labelNames[clsind], freq[clsind]))
    #
    #     for l, mm in enumerate(models):
    #         print('%s:\t%.6f' % (mm, f1[1,l,clsind]))
    #
    #
    #     print(labelNames[confmatPCA[clsind] > 0])
    #confmatPCA = np.delete(confmat, 19, axis=0)
    #print(conf)
    #conf

    axTripleBar.bar(np.arange(len(models)) - offestsTripleBar[oo]*wTripleBar, mcc[1], width=wTripleBar, yerr=mccCI[1].T, color=ccTripleBar[oo], label=omicNames[oo], align='edge', error_kw=dict(lw=1, capsize=3, capthick=1))

    axTripleBarSVM.bar(np.arange(len(models)) - offestsTripleBar[oo]*wTripleBar, mcc[0], width=wTripleBar, yerr=mccCI[0].T, color=ccTripleBar[oo], label=omicNames[oo], align='edge', error_kw=dict(lw=1, capsize=3, capthick=1))

    if oo == 0:
        axTripleBar.axhline(mcc[1,0], color='k', linestyle='--')
        axTripleBarSVM.axhline(mcc[0,0], color='k', linestyle='--')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if omic == 'omic1':
        ax = customBarPlot(w, mcc[0], mcc[1], mccCI[0].T, mccCI[1].T, ax, ['SVM', 'MLP'], modelNames, 'MCC', (0,1))
    else:
        ax = customBarPlot(w, mcc[0][:-1], mcc[1][:-1], mccCI[0][:-1].T, mccCI[1][:-1].T, ax, ['SVM', 'MLP'], modelNames[:-1], 'MCC', (0,1))
    ax.legend(loc='lower left')
    ax.set_title(omicNames[oo])

    fig.savefig('figures/rebuttal/classification_l2_atac_mcc_%s.eps' % omic, dpi=1200)
    fig.savefig('figures/rebuttal/classification_l2_atac_mcc_%s.png' % omic, dpi=600)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.scatter(f1[0,0], f1[0,1], color='C0', label='SVM')
    ax.scatter(f1[1,0], f1[1,1], color='C1', label='MLP')
    gmd = np.linspace(0,1,10)
    ax.plot(gmd, gmd, 'k--')
    ax.set_xlabel('F1 score PCA')
    ax.set_ylabel('F1 score MOFA+')
    ax.legend()

    ax = fig.add_subplot(1,2,2)
    ax.scatter(f1[0,0], f1[0,4], color='C0', label='SVM')
    ax.scatter(f1[1,0], f1[1,4], color='C1', label='MLP')

    ax.plot(gmd, gmd, 'k--')
    ax.set_xlabel('F1 score PCA')
    ax.set_ylabel('F1 score PoE')
    plt.suptitle(omicNames[oo])
    plt.tight_layout()

    plt.close(fig)
    fig.savefig('figures/rebuttal/classification_l2_atac_f1_%s.eps' % omic, dpi=1200)
    fig.savefig('figures/rebuttal/classification_l2_atac_f1_%s.png' % omic, dpi=600)

axTripleBar.set_xticks(np.arange(len(modelNames)))
axTripleBar.set_xticklabels(modelNames)
axTripleBar.legend(loc='lower left')
axTripleBar.set_ylabel('MCC', fontsize=14)

axTripleBarSVM.set_xticks(np.arange(len(modelNames)))
axTripleBarSVM.set_xticklabels(modelNames)
axTripleBarSVM.legend(loc='lower left')
axTripleBarSVM.set_ylabel('MCC', fontsize=14)

figTripleBar.savefig('figures/rebuttal/atac_mcc_tripleBar_mlp.png', dpi=600)
figTripleBar.savefig('figures/rebuttal/atac_mcc_tripleBar_mlp.eps', dpi=1200)

figTripleBarSVM.savefig('figures/rebuttal/atac_mcc_tripleBar_svm.png', dpi=600)
figTripleBarSVM.savefig('figures/rebuttal/atac_mcc_tripleBar_svm.eps', dpi=1200)

plt.close('all')



sys.exit(0)

freqInd = np.argsort(freq)
freqSorted = freq[freqInd]
cellTypes = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_cellTypes_l3.npy', allow_pickle=True)
cellTypes = np.delete(cellTypes, 19)

cellTypesSorted = cellTypes[freqInd]
f1sorted = f1[:,:, freqInd]

modelOfOrigin = []
for m in models:
    modelOfOrigin += [m] * f1sorted.shape[-1]

f1dataFrame = pd.DataFrame({'f1': f1sorted[1].flatten(), 'model': modelOfOrigin, 'label': len(models) * list(cellTypesSorted)})

fig = plt.figure()
ax = fig.add_subplot(111)
sns.barplot(x='label', y='f1', hue='model', data=f1dataFrame, ax=ax)
ax.set_xlim(46.5, 56.5)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# sns.barplot(x='label', y='f1', hue='model', data=f1dataFrame, ax=ax)



plt.close('all')
# plt.show()
