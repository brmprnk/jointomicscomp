import pickle
import numpy as np
import torch
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import pandas as pd


models = ['CGAE', 'CVAE', 'MoE', 'PoE']
modelNames = ['CGVAE', 'ccVAE', 'MoE', 'PoE']

datasets = ['GEME', 'GECNV', 'RNAATAC', 'RNAADT']

legendlocs = ['lower left', 'lower left', 'upper right', 'upper right']
abcd = ['(a)', '(b)', '(c)', '(d)']

fig = plt.figure()
i = 0
for ds in datasets:
    print(ds)
    i += 1
    # if i > 2:
    #     break

    if ds[:2] == 'GE':
        modality1 = ds[:2]
        modality2 = ds[2:]
    else:
        assert ds[:3] == 'RNA'
        modality1 = ds[:3]
        modality2 = ds[3:]

    joint = []
    m1 = []
    m2 = []
    none = []

    for m in models:
        print(m)
        with open('results/mi/%s/%s.pkl' % (ds,m), 'rb') as f:
            dd = pickle.load(f)

        ps = []
        for modality in ['x1', 'x2']:
            statistic = dd[modality + 'stats']
            null = dd[modality + 'null']

            pp = torch.sum(null >= statistic,axis=0) / null.shape[0]
            Ntests = null.shape[1]
            #pp = torch.minimum(torch.ones(Ntests), Ntests * pp)
            #pp = multipletests(np.array(pp), method='fdr_bh')[1]

            ps.append(np.array(pp))

        sig1 = ps[0] < 0.05
        sig2 = ps[1] < 0.05

        print('Number of factors: %d' % Ntests)
        x = np.sum(np.logical_and(sig1, sig2))
        print('Significant for both: %d' % x)
        joint.append(x)

        x = np.sum(np.logical_and(sig1, np.logical_not(sig2)))
        m1.append(x)
        print('Significant for modality 1 only: %d' % x)

        x = np.sum(np.logical_and(sig2, np.logical_not(sig1)))
        m2.append(x)
        print('Significant for modality 2 only: %d' % x)

        x = np.sum(np.logical_and(np.logical_not(sig1), np.logical_not(sig2)))
        none.append(x)
        print('Significant for neither: %d' % x)

        print('\n')

    ax = fig.add_subplot(2,2,i)
    df = pd.DataFrame({'joint': joint, modality1: m1, modality2: m2, 'neither': none}, index=modelNames)
    df.plot(kind='bar', stacked=True, color=['green', 'grey', 'skyblue', 'black'], ax=ax)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,2,1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=legendlocs[i-1], fontsize=5)
    #plt.legend(loc='lower left', fontsize=5)
    #
    ax.set_yticks([16*q for q in range(5)])
    plt.xticks(rotation=45)
    ax.set_ylabel('#latent factors')
    ax.set_title(abcd[i-1] +' ' +modality1 + ' + ' + modality2)

plt.tight_layout()

fig.savefig('figures/rebuttal/mires.eps', dpi=1200)
fig.savefig('figures/rebuttal/mires.png', dpi=600)

#plt.show()
