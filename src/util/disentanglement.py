import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from copy import deepcopy
import sys
from scipy.stats import spearmanr
import torch
from src.nets import MLP
from src.util.evaluate import evaluate_classification

trainInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/trainInd.npy')
validInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/validInd.npy')
testInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/testInd.npy')

cc = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/cell-cycle-phase.npy', allow_pickle=True)
tp = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/time-point.npy', allow_pickle=True)
tp = tp.astype(int)

model2folder = {'cgae': 'CGAE', 'mvib': 'MVIB', 'moe': 'MoE', 'poe': 'PoE', 'cvae': 'CVAE'}
path = 'embeddings/results/test_RNA_ADT_%s_RNA_ADT/%s/embeddings.pkl'


#
# cell-cycle
mm = []
ff = []
dt = []
pp = []
bb = []

for model in ['cgae', 'cvae', 'moe', 'poe']:

    with open(path % (model, model2folder[model]), 'rb') as f:
        embDict = pickle.load(f)


    if len(embDict['ztrain']) == 2:
        z1val = embDict['zvalidation'][0]
        z1train = embDict['ztrain'][0]
        z1test = embDict['ztest'][0]

        z2val = embDict['zvalidation'][1]
        z2train = embDict['ztrain'][1]
        z2test = embDict['ztest'][1]


        z1 = np.zeros((cc.shape[0], z1val.shape[1]))
        z1[trainInd] = z1train
        z1[validInd] = z1val
        z1[testInd] = z1test

        z2 = np.zeros((cc.shape[0], z2val.shape[1]))
        z2[trainInd] = z2train
        z2[validInd] = z2val
        z2[testInd] = z2test

        llz = [z1, z2]

    else:
        z = np.zeros((cc.shape[0], embDict['ztrain'].shape[1]))
        z[trainInd] = embDict['ztrain']
        z[validInd] = embDict['zvalidation']
        z[testInd] = embDict['ztest']

        llz = [z]


    for i, z in enumerate(llz):
        print('Cell-cycle:\t\t%s\t%d' % (model, i))

        for j in range(z.shape[1]):
            print(j)
            train = sm.add_constant(z[:, j])
            if train.shape[1] == 2:
                regressor = sm.MNLogit(cc, train, missing='drop')
                result = regressor.fit(disp=0)
                pp.append(result.pvalues[1])
                bb.append(result.params[1])
            else:
                pp.append([1., 1.])
                bb.append([0., 0.])


            mm.append(model)
            ff.append(j)
            dt.append(i)

pp = np.minimum(np.array(pp) * len(pp), 1.)
bb = np.array(bb)

ccRes = pd.DataFrame({'model': mm, 'factor': ff, 'modality': dt, 'beta G2M - G1': bb[:,0], 'beta S - G1': bb[:,1], 'p-value G2M - G1': pp[:,0], 'p-value S - G1': pp[:,1]})
# sys.exit(0)
# ---------------------------------------------------------------------------------------------------------------------------

# time-point
mm = []
ff = []
dt = []
pp = []
rr = []


for model in ['cgae', 'cvae', 'moe', 'poe']:

    with open(path % (model, model2folder[model]), 'rb') as f:
        embDict = pickle.load(f)


    if len(embDict['ztrain']) == 2:
        z1val = embDict['zvalidation'][0]
        z1train = embDict['ztrain'][0]
        z1test = embDict['ztest'][0]

        z2val = embDict['zvalidation'][1]
        z2train = embDict['ztrain'][1]
        z2test = embDict['ztest'][1]


        z1 = np.zeros((cc.shape[0], z1val.shape[1]))
        z1[trainInd] = z1train
        z1[validInd] = z1val
        z1[testInd] = z1test

        z2 = np.zeros((cc.shape[0], z2val.shape[1]))
        z2[trainInd] = z2train
        z2[validInd] = z2val
        z2[testInd] = z2test

        llz = [z1, z2]

    else:
        z = np.zeros((cc.shape[0], embDict['ztrain'].shape[1]))
        z[trainInd] = embDict['ztrain']
        z[validInd] = embDict['zvalidation']
        z[testInd] = embDict['ztest']

        llz = [z]

    for i, z in enumerate(llz):
        print('Cell-cycle:\t\t%s\t%d' % (model, i))

        for j in range(z.shape[1]):
            print(j)

            r1, p1 = spearmanr(z[:, j], tp)
            pp.append(p1)
            rr.append(r1)

            mm.append(model)
            ff.append(j)
            dt.append(i)

pp = np.minimum(np.array(pp) * len(pp), 1.)

tpRes = pd.DataFrame({'model': mm, 'factor': ff, 'modality': dt, 'rho': rr, 'p-value': pp})

# ----------------------------------------------------------------------------------------------------------------------
# print('Cell type with MLP')
# nclass = 31
#
# rna = torch.tensor(np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD.npy')[testInd]).double()
# adt = torch.tensor(np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD.npy')[testInd]).double()
#
# clf1 = MLP(rna.shape[1], 64, nclass).double()
# clf2 = MLP(adt.shape[1], 64, nclass).double()
#
# checkpoint = torch.load('type-classifier/RNA/checkpoint/model_best.pth.tar')
# clf1.load_state_dict(checkpoint['state_dict'])
#
# checkpoint = torch.load('type-classifier/ADT/checkpoint/model_best.pth.tar')
# clf2.load_state_dict(checkpoint['state_dict'])
#
# prna = clf1.predict(rna)
# padt = clf2.predict(adt)
#
# yrna = torch.argmax(prna, axis=1)
# yadt = torch.argmax(padt, axis=1)
#
#
# ytrue = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_cellType.npy')[testInd]
#
# perfRNA = evaluate_classification(ytrue, yrna)
# perfADT = evaluate_classification(ytrue, yadt)
