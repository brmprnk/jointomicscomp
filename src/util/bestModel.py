import numpy as np
import pickle
import os
import sys
import yaml

resultsDir = sys.argv[1]
configDir = sys.argv[2]


# prefix = ['tcga', 'tcga', 'tcga', 'sc']
# datasets = ['_GE_ME', '_GE_CNV', '_ME_CNV', '_RNA_ADT']
prefix = ['tcga', 'tcga', 'tcga']
datasets = ['_GE_ME', '_GE_CNV', '_ME_CNV']

# models = ['cgae', 'mvib', 'poe', 'moe', 'mofa']
# modelNames = ['CGAE', 'MVIB', 'PoE', 'MoE', 'MOFA+']
models = ['cgae', 'mvib', 'poe', 'moe']
modelNames = ['CGAE', 'MVIB', 'PoE', 'MoE']
nrCombos = {'cgae': 80, 'mvib': 240, 'poe': 80, 'moe': 160}



for pr, ds in zip(prefix, datasets):
    print(ds)
    for model, name in zip(models, modelNames):
        nc = nrCombos[model]

        bestLoss = np.inf

        for i in range(nc):
            path = resultsDir + 'train-' + pr + '-' + model + '-' + str(i) + ds + '/' + name + '/finalValidationLoss.pkl'

            try:
                with open(path, 'rb') as f:
                    res = pickle.load(f)

            except FileNotFoundError:
                print('!WARNING: Missing file %s' % path)
                continue

            if res['val_loss'] < bestLoss:
                bestModel = i
                bestLoss = res['val_loss']
                bestEpoch = res['epoch']


        print(model, bestModel, bestLoss, bestEpoch)

        with open(configDir + pr + '/' + ''.join(ds.split('_')) + '/' + model + '_' + str(bestModel) + '.yaml') as file:
            config = yaml.safe_load(file)

        config['GLOBAL_PARAMS']['task'] = 1

        config['GLOBAL_PARAMS']['name'] = 'test' + ds + '_' + model
        config[name]['pre_trained'] = resultsDir + 'train-' + pr + '-' + model + '-' + str(bestModel) + ds + '/' + name + '/checkpoint/model_epoch' + str(bestEpoch) + '.pth.tar'



        with open(configDir + 'best-models/' + ''.join(ds.split('_')) + '_' + model + '.yaml', 'w') as writeConf:
            yaml.safe_dump(config, writeConf)