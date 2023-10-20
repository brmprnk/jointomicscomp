import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from src.baseline.baseline import trainRegressor, classification, classificationMLP
from src.nets import OmicRegressor, OmicRegressorSCVI
import torch
from src.util import logger
from src.CGAE.model import evaluateUsingBatches, MultiOmicsDataset, evaluatePerDatapoint
from torch.utils.data import DataLoader
from src.util.early_stopping import EarlyStopping
import pickle


trainFactorsFile = 'src/MCIA/RNAATAC_train_factors_20.csv'

data_path1 = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/rna.npy'
data_path2 = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/atac.npy'

train_ind_file = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/trainInd.npy'
val_ind_file = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/validInd.npy'
test_ind_file = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/testInd.npy'

sample_namesFile = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/sampleNames.npy'


ztrainDF = pd.read_csv(trainFactorsFile, index_col=0)
ztrain = np.array(ztrainDF)

X1 = np.load(data_path1)
X2 = np.load(data_path2)

sampleNames = np.load(sample_namesFile, allow_pickle=True)
trainInd = np.load(train_ind_file)
validInd = np.load(val_ind_file)
testInd = np.load(test_ind_file)

X1train = X1[trainInd]
X2train = X2[trainInd]


mapper1 = LinearRegression()
mapper1.fit(X1train, ztrain)
print('Projection from modality 1 to z\nMean R^2 on training data: %.4f' % mapper1.score(X1train, ztrain))


mapper2 = LinearRegression()
mapper2.fit(X2train, ztrain)
print('Projection from modality 2 to z\nMean R^2 on training data: %.4f' % mapper2.score(X2train, ztrain))

xconcatTrain = np.hstack((X1train, X2train))

mapperJoint = LinearRegression()
mapperJoint.fit(xconcatTrain, ztrain)
print('Projection from both modalities to z\nMean R^2 on training data: %.4f' % mapperJoint.score(xconcatTrain, ztrain))


dirName = 'embeddings/results/test_RNA_ATAC_mcia_RNA_ATAC/MCIA/'
os.makedirs(dirName, exist_ok=True)


embDict = dict()
embDict['ztrain'] = [mapper1.predict(X1train), mapper2.predict(X2train), ztrain]

X1valid = X1[validInd]
X2valid = X2[validInd]
XvalidConcat = np.hstack((X1valid, X2valid))
embDict['zvalidation'] = [mapper1.predict(X1valid), mapper2.predict(X2valid), mapperJoint.predict(XvalidConcat)]


X1test = X1[testInd]
X2test = X2[testInd]
XtestConcat = np.hstack((X1test, X2test))
embDict['ztest'] = [mapper1.predict(X1test), mapper2.predict(X2test), mapperJoint.predict(XtestConcat)]


# save embeddings
with open(dirName + 'embeddings.pkl', 'wb') as f:
    pickle.dump(embDict, f)


# train imputation model from ztrain to X1 and to X2

logger.output_file = 'src/MCIA/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info("Selected device: {}".format(device))
torch.manual_seed(1)

save_dir = 'src/MCIA/imputationATACfromRNA'
os.makedirs(save_dir, exist_ok=True)
ckpt_dir = save_dir + '/checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
logs_dir = save_dir + '/logs'


likelihoods = ['nb', 'bernoulli']
n_categories = [-1, -1]

# Initialize GLMs
net2from1 = OmicRegressor(ztrain.shape[1], X2train.shape[1], distribution=likelihoods[1], optimizer_name='Adam', lr=0.0001, n_categories=n_categories[1])
net2from1 = net2from1.to(device).double()

net1from2 = OmicRegressorSCVI(ztrain.shape[1], X1train.shape[1], distribution=likelihoods[0], optimizer_name='Adam', lr=0.0001, use_batch_norm=False, log_input=False)
net1from2 = net1from2.to(device).double()


# modality 2 from 1
dataTrain = [torch.tensor(ztrain, device=device), torch.tensor(X2train, device=device)]
dataValidation = [torch.tensor(embDict['zvalidation'][0], device=device), torch.tensor(X2valid, device=device)]
dataTest = [torch.tensor(embDict['ztest'][0], device=device), torch.tensor(X2test, device=device)]

datasetTrain = MultiOmicsDataset(dataTrain)
datasetValidation = MultiOmicsDataset(dataValidation)
datasetTest = MultiOmicsDataset(dataTest)

validationEvalBatchSize = 64
trainEvalBatchSize = 64


train_loader = DataLoader(datasetTrain, batch_size=64, shuffle=True, num_workers=0,
                          drop_last=False)

train_loader_eval = DataLoader(datasetTrain, batch_size=trainEvalBatchSize, shuffle=False, num_workers=0,
                               drop_last=False)

valid_loader = DataLoader(datasetValidation, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                          drop_last=False)

test_loader = DataLoader(datasetTest, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                          drop_last=False)

test_loader_individual = DataLoader(datasetTest, batch_size=1, shuffle=False, num_workers=0,
                          drop_last=False)


early_stopping = EarlyStopping(patience=10, verbose=True)

bestLoss, bestEpoch = trainRegressor(device=device, net=net2from1, num_epochs=500, train_loader=train_loader,
      train_loader_eval=train_loader_eval, valid_loader=valid_loader,
      ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping)


logger.info("Using model from epoch %d" % bestEpoch)

checkpoint = torch.load(ckpt_dir + '/model_best.pth.tar')

net2from1.load_state_dict(checkpoint['state_dict'])


metricsValidation = evaluateUsingBatches(net2from1, device, valid_loader, True)
metricsTest = evaluateUsingBatches(net2from1, device, test_loader, True)

logger.info('Validation performance, imputation error modality 2 from 1')
for m in metricsValidation:
    logger.info('%s\t%.4f' % (m, metricsValidation[m]))

logger.info('Test performance, imputation error modality 2 from 1')
for m in metricsTest:
    logger.info('%s\t%.4f' % (m, metricsTest[m]))

metricsTestIndividual2from1 = evaluatePerDatapoint(net2from1, device, test_loader_individual, True)


# modality 1 from 2

save_dir = 'src/MCIA/imputationRNAfromATAC'
os.makedirs(save_dir, exist_ok=True)
ckpt_dir = save_dir + '/checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
logs_dir = save_dir + '/logs'


dataTrain = [torch.tensor(ztrain, device=device), torch.tensor(X1train, device=device)]
dataValidation = [torch.tensor(embDict['zvalidation'][1], device=device), torch.tensor(X1valid, device=device)]
dataTest = [torch.tensor(embDict['ztest'][1], device=device), torch.tensor(X1test, device=device)]

datasetTrain = MultiOmicsDataset(dataTrain)
datasetValidation = MultiOmicsDataset(dataValidation)
datasetTest = MultiOmicsDataset(dataTest)

validationEvalBatchSize = 64
trainEvalBatchSize = 64


train_loader = DataLoader(datasetTrain, batch_size=64, shuffle=True, num_workers=0,
                          drop_last=False)

train_loader_eval = DataLoader(datasetTrain, batch_size=trainEvalBatchSize, shuffle=False, num_workers=0,
                               drop_last=False)

valid_loader = DataLoader(datasetValidation, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                          drop_last=False)

test_loader = DataLoader(datasetTest, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                          drop_last=False)

test_loader_individual = DataLoader(datasetTest, batch_size=1, shuffle=False, num_workers=0,
                          drop_last=False)

early_stopping = EarlyStopping(patience=10, verbose=True)

bestLoss, bestEpoch = trainRegressor(device=device, net=net1from2, num_epochs=500, train_loader=train_loader,
      train_loader_eval=train_loader_eval, valid_loader=valid_loader,
      ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping)


logger.info("Using model from epoch %d" % bestEpoch)

checkpoint = torch.load(ckpt_dir + '/model_best.pth.tar')

net1from2.load_state_dict(checkpoint['state_dict'])


metricsValidation = evaluateUsingBatches(net1from2, device, valid_loader, True)
metricsTest = evaluateUsingBatches(net1from2, device, test_loader, True)

logger.info('Validation performance, imputation error modality 2 from 1')
for m in metricsValidation:
    logger.info('%s\t%.4f' % (m, metricsValidation[m]))

logger.info('Test performance, imputation error modality 2 from 1')
for m in metricsTest:
    logger.info('%s\t%.4f' % (m, metricsTest[m]))

metricsTestIndividual1from2 = evaluatePerDatapoint(net1from2, device, test_loader_individual, True)

individualPerfomanceSave = 'src/MCIA/imputation_RNAATAC_performancePerDatapont.pkl'

with open(individualPerfomanceSave, 'wb') as f:
    pickle.dump({'1from2': metricsTestIndividual1from2, '2from1': metricsTestIndividual2from1}, f)


#############################
# task 2 cell type classification

classLabels = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/celltype_l2.npy')
labelNames = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/celltypes_l2.npy', allow_pickle=True)

ytrain = classLabels[trainInd]
yvalid = classLabels[validInd]
ytest = classLabels[testInd]

_, acc, pr, rc, f1, mcc, confMat, CIs = classification(embDict['ztrain'][0], ytrain, embDict['zvalidation'][0], yvalid, embDict['ztest'][0], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
performance1 = [acc, pr, rc, f1, mcc, confMat]

logger.info('Test performance, classification task, linear classifier, modality 1')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

_, acc, pr, rc, f1, mcc, confMat, CIs = classification(embDict['ztrain'][1], ytrain, embDict['zvalidation'][1], yvalid, embDict['ztest'][1], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
performance2 = [acc, pr, rc, f1, mcc, confMat]

logger.info('Test performance, classification task, linear classifier, modality 2')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}


_, acc, pr, rc, f1, mcc, confMat, CIs = classification(embDict['ztrain'][2], ytrain, embDict['zvalidation'][2], yvalid, embDict['ztest'][2], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
performance12 = [acc, pr, rc, f1, mcc, confMat]

logger.info('Test performance, classification task, linear classifier, both modalities')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}

level = 'l2'

# -----------------------------------------------------------------
_, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(embDict['ztrain'][0], ytrain, embDict['zvalidation'][0], yvalid, embDict['ztest'][0], ytest, 'type-classifier/eval/' + level + '/mcia_RNA/')
performance1 = [acc, pr, rc, f1, mcc, confMat]

logger.info('Test performance, classification task, non-linear classifier, modality 1')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


mlp_pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}


_, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(embDict['ztrain'][1], ytrain, embDict['zvalidation'][1], yvalid, embDict['ztest'][1], ytest, 'type-classifier/eval/' + level + '/mcia_ADT/')
performance2 = [acc, pr, rc, f1, mcc, confMat]

logger.info('Test performance, classification task, non-linear classifier, modality 2')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

mlp_pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

_, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(embDict['ztrain'][2], ytrain, embDict['zvalidation'][2], yvalid, embDict['ztest'][2], ytest, 'type-classifier/eval/' + level + '/mcia_RNA_ADT/')
performance12 = [acc, pr, rc, f1, mcc, confMat]

logger.info('Test performance, classification task, non-linear classifier, both modalities')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

mlp_pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}

save_dir = 'src/MCIA/'
ext = 'atac_'

logger.info("Saving results")


with open(save_dir + "/" + ext + "MCIA_task2_results.pkl", 'wb') as f:
    pickle.dump({'omic1': pr1, 'omic2': pr2, 'omic1+2': pr12, 'omic1-mlp': mlp_pr1, 'omic2-mlp': mlp_pr2, 'omic1+2-mlp': mlp_pr12}, f)
