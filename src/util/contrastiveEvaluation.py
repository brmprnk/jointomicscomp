import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, top_k_accuracy_score


models = ['mcia', 'mofa', 'cgae', 'moe']
modelDirs = ['MCIA', 'MOFA+', 'CGAE', 'MoE']


datasets = ['GE_ME', 'GE_CNV', 'RNA_ADT']

kk = ['ztrain', 'zvalidation', 'ztest']


for i, (m, mn) in enumerate(zip(models, modelDirs)):
    print(mn)
    for j, ds in enumerate(datasets):

        if m == 'mcia' and ds == 'RNA_ADT':
            continue

        print(ds)
        try:
            with open('embeddings/results/test_%s_%s_%s/%s/embeddings.pkl' % (ds, m, ds, mn), 'rb') as f:
                dd = pickle.load(f)
        except FileNotFoundError:
            # assert i == 1
            # assert j == 2
            with open('embeddings/results/test_%s_%s_%s/MOFA/embeddings.pkl' % (ds, m, ds), 'rb') as f:
                dd = pickle.load(f)


        for tvt in kk:
            print(tvt)

            z1 = dd[tvt][0]
            z2 = dd[tvt][1]

            assert z1.shape ==z2.shape

            try:
                DD = cdist(z1, z2)
            except MemoryError:
                print('Skipping... Too big')
                continue

            modalities = ds.split('_')
            print('NN of %s profile among %s profiles' % (modalities[1], modalities[0]))
            print('Top-1: %.3f' % accuracy_score(np.arange(z1.shape[0]), np.argmin(DD, axis=0)))
            print('Top-5: %.3f' % top_k_accuracy_score(np.arange(z1.shape[0]), -DD.T, k=2))
            print('\n')
            print('NN of %s profile among %s profiles' % (modalities[0], modalities[1]))
            print('Top-1: %.3f' % accuracy_score(np.arange(z1.shape[0]), np.argmin(DD, axis=1)))
            print('Top-5: %.3f' % top_k_accuracy_score(np.arange(z1.shape[0]), -DD, k=5))

            print('\n')

        print('\n\n')

ns = set()
for j, ds in enumerate(datasets):
    print(ds)

    with open('embeddings/results/test_%s_%s_%s/%s/embeddings.pkl' % (ds, m, ds, mn), 'rb') as f:
        dd = pickle.load(f)

    for tvt in kk:
        ns.add(dd[tvt][0].shape[0])

# ignoring training set of sc due to size
ns = sorted(ns)[:-1][::-1]
niter = 100

np.random.seed(1)
top1 = np.zeros((len(ns), niter))
top5 = np.zeros((len(ns), niter))
for i, n in enumerate(ns):
    print(i)
    x = np.arange(n)
    for j in range(niter):
        print(n,j)
        posteriors = np.random.rand(n,n)

        top5[i,j] = top_k_accuracy_score(x, posteriors)
        # doesn't matter which axis, should be the same either way
        pred = np.argmin(posteriors, axis=0)
        top1[i,j] = accuracy_score(x, pred)
