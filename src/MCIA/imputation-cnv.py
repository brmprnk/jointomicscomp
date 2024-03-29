import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from src.baseline.baseline import trainRegressor
from src.nets import OmicRegressor
import torch
from src.util import logger
from src.CGAE.model import evaluateUsingBatches, MultiOmicsDataset, evaluatePerDatapoint
from torch.utils.data import DataLoader
from src.util.early_stopping import EarlyStopping
import pickle

trainFactorsFile = 'src/MCIA/gecnv_train_factors_16.csv'

data_path1 = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/GE.npy'
data_path2 = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/CNV.npy'

test_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/testInd.npy'
train_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/trainInd.npy'
val_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/validInd.npy'
sample_namesFile = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/sampleNames.npy'


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


dirName = 'embeddings/results/test_GE_CNV_mcia_GE_CNV/MCIA/'
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

save_dir = 'src/MCIA/imputationCNVfromGE'
os.makedirs(save_dir, exist_ok=True)
ckpt_dir = save_dir + '/checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
logs_dir = save_dir + '/logs'


likelihoods = ['normal', 'categorical']
n_categories = [-1, 5]

# Initialize GLMs
net2from1 = OmicRegressor(ztrain.shape[1], X2train.shape[1], distribution=likelihoods[1], optimizer_name='Adam', lr=0.0001, n_categories=n_categories[1])
net2from1 = net2from1.to(device).double()

net1from2 = OmicRegressor(ztrain.shape[1], X1train.shape[1], distribution=likelihoods[0], optimizer_name='Adam', lr=0.0001, n_categories=n_categories[0])
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

save_dir = 'src/MCIA/imputationGEfromCNV'
os.makedirs(save_dir, exist_ok=True)
ckpt_dir = save_dir + '/checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
logs_dir = save_dir + '/logs'


#some times inferences of zs goes wrong and values become huge. remove those samples
dataTrain = [torch.tensor(ztrain, device=device), torch.tensor(X2train, device=device)]

norm = np.sum(embDict['zvalidation'][1] ** 2, axis=1)
ii = np.where(norm < 50)[0]
dataValidation = [torch.tensor(embDict['zvalidation'][1][ii], device=device), torch.tensor(X1valid[ii], device=device)]

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

try:
    checkpoint = torch.load(ckpt_dir + '/model_best.pth.tar')
except FileNotFoundError:
    logger.info('Training did not complete properly, using last epoch')
    checkpoint = torch.load(ckpt_dir + '/model_last.pth.tar')


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

individualPerfomanceSave = 'src/MCIA/imputation_GECNV_performancePerDatapont.pkl'

with open(individualPerfomanceSave, 'wb') as f:
    pickle.dump({'1from2': metricsTestIndividual1from2, '2from1': metricsTestIndividual2from1}, f)
