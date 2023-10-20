import numpy as np
import pickle
import os
import sys
import yaml

resultsDir = sys.argv[1]
configDir = sys.argv[2]



nrCombos = {'cgae': 80, 'poe': 80, 'moe': 160, 'cvae':80, 'totalvi': 64, 'uniport': 16}

models = ['cgae', 'cvae', 'moe', 'poe']
modelNames = ['CGAE', 'CVAE', 'MoE', 'PoE']
datasets = ['_GE_ME', '_GE_CNV', '_RNA_ADT', '_RNA_ATAC']
prefix = ['tcga', 'tcga', 'cite', 'atac']

prefix = ['atac']
datasets = ['_RNA_ATAC']


for pr, ds in zip(prefix, datasets):
    print(ds)
    for model, name in zip(models, modelNames):
        nc = nrCombos[model]

        bestLoss = np.inf

        for i in range(nc):
            #path = resultsDir + 'train-' + pr + '-' + model + '-' + str(i) + ds + '/' + name + '/finalValidationLoss.pkl'
            if model != 'totalvi':
                path = resultsDir + 'likelihood-' + pr + '-' + model + '-' + str(i) + ds + '/' + name + '/finalValidationLoss.pkl'
            else:
                path = resultsDir + 'likelihood-' + pr + '-' + model + '-test-' + str(i) + ds + '/' + name + '/finalValidationLoss.pkl'

            try:
                with open(path, 'rb') as f:
                    res = pickle.load(f)

            except FileNotFoundError:
                print('!WARNING: Missing file %s' % path)
                continue
            #print(model, i, res['val_loss'], res['epoch'])
            if res['val_loss'] < bestLoss:
                bestModel = i
                bestLoss = res['val_loss']
                bestEpoch = res['epoch']

            print('%s: model %d, epoch: %d, loss:%.2f' %(name, i, res['epoch'], res['val_loss']))
        print(model, bestModel, bestLoss, bestEpoch)

        try:
            with open(configDir + pr + '/' + ''.join(ds.split('_')) + '/' + model + '_' + str(bestModel) + '.yaml') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            assert pr == 'cite' or pr == 'atac'
            assert ds == '_RNA_ADT' or ds == '_RNA_ATAC'

            if pr == 'cite':
                extra = 'sc/'
            else:
                extra = 'atac/'

            if model != 'totalvi':
                with open(configDir + extra + model + '_' + str(bestModel) + '.yaml') as file:
                    config = yaml.safe_load(file)
            else:
                with open(configDir + extra + name + '_' + str(bestModel) + '.yaml') as file:
                    config = yaml.safe_load(file)


        if pr == 'tcga':
            # for tcga only do imputation and save embeddings
            config['GLOBAL_PARAMS']['task'] = 1
        else:
            # for cite-seq additionally do cell type classification
            config['GLOBAL_PARAMS']['task'] = 2
            config[name]['clf_criterion'] = 'mcc'

        config['GLOBAL_PARAMS']['name'] = 'test' + ds + '_' + model

        if model != 'totalvi':
            config[name]['pre_trained'] = resultsDir + 'likelihood-' + pr + '-' + model + '-' + str(bestModel) + ds + '/' + name + '/checkpoint/model_best.pth.tar'
        else:
            config[name]['pre_trained'] = resultsDir + 'likelihood-' + pr + '-' + model + '-test-' + str(bestModel) + ds + '/' + name + '/checkpoint/'


        with open(configDir + 'best-models/' + ''.join(ds.split('_')) + '_' + model + '.yaml', 'w') as writeConf:
            yaml.safe_dump(config, writeConf)


    # uniport
    model = 'uniport'
    name = 'UNIPORT'
    nc = nrCombos[model]
    path = 'results/uniport/' + pr + ds + '/'

    outfiles = os.listdir(path)
    nrs = sorted([int(k.split('.')[0].split('-')[1]) for k in outfiles])
    outfiles = ['slurm-%d.out' % k for k in nrs]
    bestLoss = np.inf
    bestModel = -1
    for i in range(nc):
        with open(path + outfiles[i], 'r') as f:
            ll = f.readlines()[-1].split(',')

        rec = float(ll[2].split('=')[-1])
        kl = float(ll[-1].split('=')[-1].split(']')[0])

        train_loss = rec + kl
        if train_loss < bestLoss:
            bestModel = i
            bestLoss = train_loss


    print(model, bestModel, bestLoss)
    try:
        with open(configDir + pr + '/' + ''.join(ds.split('_')) + '/' + model + '_' + str(bestModel) + '.yaml') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        assert pr == 'cite' or pr == 'atac'
        assert ds == '_RNA_ADT' or ds == '_RNA_ATAC'

        if pr == 'cite':
            extra = 'sc/'
        else:
            extra = 'atac/'

        with open(configDir + extra + model + '_' + str(bestModel) + '.yaml') as file:
            config = yaml.safe_load(file)

    if pr == 'tcga':
        # for tcga only do imputation and save embeddings
        config['GLOBAL_PARAMS']['task'] = 1
    else:
        # for cite-seq additionally do cell type classification
        config['GLOBAL_PARAMS']['task'] = 2
        config[name]['clf_criterion'] = 'mcc'

    config['GLOBAL_PARAMS']['name'] = 'test' + ds + '_' + model
    config[name]['pre_trained'] = 'results/uniport/results/' + pr + '-uniport-' + str(bestModel) + ds + '/UniPort/checkpoint/'



    with open(configDir + 'best-models/' + ''.join(ds.split('_')) + '_' + model + '.yaml', 'w') as writeConf:
        yaml.safe_dump(config, writeConf)
