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

# resultDict = dict()
# resultDict['GE_ME'] = {
# 'GLM': 'embeddings/results/test-baseline_GE_ME/baseline/test-baselinetest_performance_per_datapoint.pkl',
# 'MOFA+': 'src/MOFA2/imputation_GEME_performancePerDatapont.pkl',
# 'MCIA': 'src/MCIA/imputation_GEME_performancePerDatapont.pkl',
# 'CGAE': 'embeddings/results/test_GE_ME_cgae_GE_ME/CGAE/test_performance_per_datapoint.pkl',
# 'CVAE': 'embeddings/results/test_GE_ME_cvae_GE_ME/CVAE/test_performance_per_datapoint.pkl',
# 'MoE': 'embeddings/results/test_GE_ME_moe_GE_ME/MoE/test_performance_per_datapoint.pkl',
# 'PoE': 'embeddings/results/test_GE_ME_poe_GE_ME/PoE/test_performance_per_datapoint.pkl',
# 'labelnames': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerTypes.npy',
# 'labels': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerType.npy',
# 'test_ind': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/testInd.npy'}
#
#
# resultDict['GE_CNV'] = {
# 'GLM': 'embeddings/results/likelihood-tcga-baseline-GECNV_GE_CNV/baseline/likelihood-tcga-baseline-GECNVtest_performance_per_datapoint.pkl',
# 'MOFA+': 'src/MOFA2/imputation_GECNV_performancePerDatapont.pkl',
# 'MCIA': 'src/MCIA/imputation_GECNV_performancePerDatapont.pkl',
# 'CGAE': 'embeddings/results/test_GE_CNV_cgae_GE_CNV/CGAE/test_performance_per_datapoint.pkl',
# 'CVAE': 'embeddings/results/test_GE_CNV_cvae_GE_CNV/CVAE/test_performance_per_datapoint.pkl',
# 'MoE': 'embeddings/results/test_GE_CNV_moe_GE_CNV/MoE/test_performance_per_datapoint.pkl',
# 'PoE': 'embeddings/results/test_GE_CNV_poe_GE_CNV/PoE/test_performance_per_datapoint.pkl',
# 'labelnames': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerTypes.npy',
# 'labels': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/cancerType.npy',
# 'test_ind': '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/testInd.npy'}
#
# resultDict['RNA_ADT'] = {
# 'GLM': 'embeddings/results/test-cite-baseline-l3_RNA_ADT/baseline/test-cite-baseline-l3test_performance_per_datapoint.pkl',
# 'MOFA+': 'src/MOFA2/imputation_RNAADT_performancePerDatapont.pkl',
# 'MCIA': '',
# 'CGAE': 'embeddings/results/test_RNA_ADT_cgae_RNA_ADT/CGAE/test_performance_per_datapoint.pkl',
# 'CVAE': 'embeddings/results/test_RNA_ADT_cvae_RNA_ADT/CVAE/test_performance_per_datapoint.pkl',
# 'MoE': 'embeddings/results/test_RNA_ADT_moe_RNA_ADT/MoE/test_performance_per_datapoint.pkl',
# 'PoE': 'embeddings/results/test_RNA_ADT_poe_RNA_ADT/PoE/test_performance_per_datapoint.pkl',
# 'labelnames': '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD_cellTypes.npy',
# 'labels': '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD_cellType.npy',
# 'test_ind': '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/testInd.npy'}
#
#
# w = 0.4
# models = ['GLM', 'MCIA', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE']
# #modelNames = ['cgae', 'cvae', 'poe', 'moe']
#
# datasets = ['GE_ME', 'GE_CNV', 'RNA_ADT']
#
# fwerCorr = binom(len(models), 2)
#
# myylims = [(-10000, 10000), (-30000, 3000), (-2500, 0)]
# myvmins = [[-12308.2295, 0], [-20000, 15000], [-1571.9639, -952.5779]]
#
#
#
# for i, ds in enumerate(datasets):
#     modalities = ds.split('_')
#
#
#     labels = ['%s from %s' % (modalities[0], modalities[1]), '%s from %s' % (modalities[1], modalities[0])]
#
#     obs1 = []
#     obs2 = []
#
#     if i == 2:
#         models = ['GLM', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE']
#
#     ll1 = np.zeros(len(models))
#     ll2 = np.zeros(len(models))
#
#     std1 = np.zeros(len(models))
#     std2 = np.zeros(len(models))
#     fwerCorr = binom(len(models), 2)
#
#     labelsAll = np.load(resultDict[ds]['labels'])
#     labelNames = np.load(resultDict[ds]['labelnames'])
#     tstInd = np.load(resultDict[ds]['test_ind'])
#
#     labels = labelsAll[tstInd]
#     labels = list(pd.Series(labels).map(pd.Series(labelNames).to_dict()))
#
#
#     for j, model in enumerate(models):
#         with open(resultDict[ds][model], 'rb') as f:
#              dd = pickle.load(f)
#
#         if 'LL1/2' in dd.keys():
#             key12 = 'LL1/2'
#             key21 = 'LL2/1'
#
#             # n = dd[key12].shape[0]
#
#             numbers = dd[key12].numpy()
#             if model == 'PoE':
#                 numbers *= -1
#                 # by mistake in per datapoint eval of poe the loss is returned
#
#             obs1.append(numbers)
#             ll1[j] = np.mean(numbers)
#             std1[j] = np.std(numbers, ddof=1)
#
#             numbers = dd[key21].numpy()
#             if model == 'PoE':
#                 numbers *= -1
#                 # by mistake in per datapoint eval of poe the loss is returned
#
#             obs2.append(numbers)
#             ll2[j] = np.mean(numbers)
#             std2[j] = np.std(numbers, ddof=1)
#
#
#         else:
#             key12 = '1from2'
#             key21 = '2from1'
#
#             obs1.append(-1 * dd[key12]['loss'].numpy())
#             ll1[j] = np.mean(-1 * dd[key12]['loss'].numpy())
#             std1[j] = np.std(-1 * dd[key12]['loss'].numpy(), ddof=1)
#
#             obs2.append(-1 * dd[key21]['loss'].numpy())
#             ll2[j] = np.mean(-1 * dd[key21]['loss'].numpy())
#             std2[j] = np.std(-1 * dd[key21]['loss'].numpy(), ddof=1)
#
#
#
#     rnaImp = np.array(obs1).flatten()
#     adtImp = np.array(obs2).flatten()
#     mmm = []
#     for m in models:
#         mmm += [m] * len(obs1[0])
#
#     mmm = mmm + mmm
#     label = labels * (2 * len(models))
#
#     modality = [modalities[0] + ' from ' + modalities[1]] * len(rnaImp)
#     modality += [modalities[1] + ' from ' + modalities[0]] * len(adtImp)
#
#     hugeDF = pd.DataFrame({'Log likelihood': np.hstack((rnaImp, adtImp)), 'Model': mmm, 'modality': modality, 'label': label})
#
#     #
#     # print('ANOVA, rec of modality 1')
#     # stat, p = friedmanchisquare(obs1)
#     # print('p = %.20f' % p)
#     #
#     #
#     # print('ANOVA, rec of modality 2')
#     # stat, p = friedmanchisquare(obs2)
#     #
#     # print('p = %.20f' % p)
#     #
#     # print('Post hoc tests')
#     # for kk in range(len(models) - 1):
#     #     for ll in range(kk + 1, len(models)):
#     #         pval = wilcoxon(obs1[kk], obs1[ll], correction=True)[1] * fwerCorr
#     #         print('Modality 1, %s vs %s: p = %.20f' % (models[kk], models[ll], pval))
#     #
#     #         pval = wilcoxon(obs2[kk], obs2[ll], correction=True)[1] * fwerCorr
#     #         print('Modality 2, %s vs %s: p = %.20f' % (models[kk], models[ll], pval))
#
#
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     #
#     # ax = customBarPlot(w, ll1, ll2, std1, std2, ax, labels, models, ylim=myylims[i])
#
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#
#     sns.boxplot(x='Model', y='Log likelihood', hue='modality', data=hugeDF, ax=ax)
#     means = np.array(hugeDF[hugeDF['Model'] == 'GLM'].groupby('modality').median()['Log likelihood']).reshape(2,)
#
#     if i == 0:
#         ax.axhline(means[0], color='C0', linestyle='--', linewidth=1, alpha=0.5)
#         ax.axhline(means[1], color='C1', linestyle='--', linewidth=1, alpha=0.5)
#     else:
#         ax.axhline(means[1], color='C0', linestyle='--', linewidth=1, alpha=0.5)
#         ax.axhline(means[0], color='C1', linestyle='--', linewidth=1, alpha=0.5)
#
#     ax.set_ylim(myylims[i])
#     # plt.grid()
#     ax.tick_params(axis='y',labelsize=8)
#     # fig.savefig('figures/imputation%s_%s.eps' % (modalities[0], modalities[1]), dpi=1200)
#
#     # fig2 = plt.figure()
#     # gmd = twoBarPlot2(w, ll1, ll2, std1, std2, fig2, labels, models)
#
#     medianModality1PerClass = hugeDF[hugeDF['modality'] == modality[0]].groupby(['Model', 'label']).median().values.reshape(len(models), np.unique(labelNames).shape[0])
#     medianModality2PerClass = hugeDF[hugeDF['modality'] == modality[-1]].groupby(['Model', 'label']).median().values.reshape(len(models), np.unique(labelNames).shape[0])
#
#     zmodel = linkage(medianModality1PerClass, method='average', metric='correlation')
#     zctype = linkage(medianModality1PerClass.T, method='average', metric='euclidean')
#
#     ff = sns.clustermap(pd.DataFrame(medianModality1PerClass, index=sorted(models), columns=sorted(labelNames)), cmap='Greens', row_linkage=zmodel, col_linkage=zctype, vmin=myvmins[i][0])
#     plt.tight_layout()
#     ff.savefig('figures/imputationPerClass_%s_%s.eps' % (modalities[0], modalities[1]), dpi=1200)
#
#     zmodel = linkage(medianModality2PerClass, method='average', metric='correlation')
#     zctype = linkage(medianModality2PerClass.T, method='average', metric='euclidean')
#
#     ff = sns.clustermap(pd.DataFrame(medianModality2PerClass, index=sorted(models), columns=sorted(labelNames)), cmap='Greens', row_linkage=zmodel, col_linkage=zctype, vmin=myvmins[i][0])
#     plt.tight_layout()
#     ff.savefig('figures/imputationPerClass_%s_%s.eps' % (modalities[1], modalities[0]), dpi=1200)
#
#
#
# plt.close('all')
# # plt.show()
# sys.exit(0)
resultDictClf = dict()
resultDictClf['RNA_ADT'] = {
'PCA': 'embeddings/results/test-cite-baseline-l3_RNA_ADT/baseline/test-cite-baseline-l3results_pickle',
'MOFA+': 'src/MOFA2/MOFA_task2_results.pkl',
'MCIA': '',
'CGAE': 'embeddings/results/test_RNA_ADT_cgae_RNA_ADT/CGAE/CGAE_task2_results.pkl',
'CVAE': 'embeddings/results/test_RNA_ADT_cvae_RNA_ADT/CVAE/CVAE_task2_results.pkl',
'MoE': 'embeddings/results/test_RNA_ADT_moe_RNA_ADT/MoE/CGAE_task2_results.pkl',
'PoE': 'embeddings/results/test_RNA_ADT_poe_RNA_ADT/PoE/PoE_task2_results.pkl'}

labelNames = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_cellTypes_l3.npy', allow_pickle=True)
labelNames = np.delete(labelNames, 19)
models = ['PCA', 'MOFA+', 'CGAE', 'CVAE', 'PoE', 'MoE']
#modelNames = ['cgae', 'cvae', 'poe', 'moe']

ds = 'RNA_ADT'


modalities = ds.split('_')
mcc = np.zeros((2,len(models)))
f1 = np.zeros((2, len(models), 57))

mccCI = np.zeros((2, len(models), 2))
#f1CI = np.zeros((len(models), 57,2))

freq = np.load('l3_class_frequencies_test_data.npy')



omicNames = ['RNA+ADT', 'RNA', 'ADT']
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
                mcc[k,j] = dd[omic + '-mlp']['mcc']
                mccCI[k,j] = dd[omic + '-mlp']['CIs'][:,-1]

                mccCI[k, j, 0] = mcc[k, j] - mccCI[k, j,0]
                mccCI[k, j, 1] = mccCI[k, j,1] - mcc[k, j]

                tmp = dd[omic + '-mlp']['f1']


            if tmp.shape[0] == 57:
                f1[k,j] = tmp
            else:
                # this model made predictions for tye class not in the test set
                f1[k,j] = np.delete(tmp, 19)

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
    if confmatPCA.shape != (57, 57):
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
    break




    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax = customBarPlot(w, mcc[0], mcc[1], mccCI[0].T, mccCI[1].T, ax, ['SVM', 'MLP'], models, 'MCC', (0,1))
    # ax.legend(loc='lower left')
    # ax.set_title(omicNames[oo])
    #
    # fig.savefig('figures/classification_l3_mcc_%s.eps' % omic, dpi=1200)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1,2,1)
    # ax.scatter(f1[0,0], f1[0,1], color='C0', label='SVM')
    # ax.scatter(f1[1,0], f1[1,1], color='C1', label='MLP')
    # gmd = np.linspace(0,1,10)
    # ax.plot(gmd, gmd, 'k--')
    # ax.set_xlabel('F1 score PCA')
    # ax.set_ylabel('F1 score MOFA+')
    # ax.legend()
    #
    # ax = fig.add_subplot(1,2,2)
    # ax.scatter(f1[0,0], f1[0,4], color='C0', label='SVM')
    # ax.scatter(f1[1,0], f1[1,4], color='C1', label='MLP')
    #
    # ax.plot(gmd, gmd, 'k--')
    # ax.set_xlabel('F1 score PCA')
    # ax.set_ylabel('F1 score PoE')
    # plt.suptitle(omicNames[oo])
    # plt.tight_layout()
    #
    # fig.savefig('figures/classification_l3_f1_%s.eps' % omic, dpi=1200)
    #

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
