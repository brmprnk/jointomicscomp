import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from src.baseline.baseline import trainRegressor, classification, classificationMLP
from src.nets import OmicRegressor, OmicRegressorSCVI
import torch
from src.util import logger
from src.util.evaluate import evaluate_generation
from src.CGAE.model import evaluateUsingBatches, MultiOmicsDataset, evaluatePerDatapoint
from torch.utils.data import DataLoader
from src.util.early_stopping import EarlyStopping
import pickle
import sys

task = int(sys.argv[1])



try:
    omic1 = sys.argv[2]
    omic2 = sys.argv[3]



    if omic1 != 'RNA':
        trainFactorsFile = 'src/MOFA2/mofa_tcga_factors_%s%s_training.tsv' % (omic1, omic2)

        data_path1 = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/%s.npy' % omic1
        data_path2 = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/%s.npy' % omic2

        train_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/trainInd.npy'
        val_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/validInd.npy'
        test_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/testInd.npy'

        sample_namesFile = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/sampleNames.npy'

        weightsFile1 = 'src/MOFA2/mofa_tcga_weights_%s%s_%s.tsv' % (omic1, omic2, omic1)
        weightsFile2 = 'src/MOFA2/mofa_tcga_weights_%s%s_%s.tsv' % (omic1, omic2, omic2)

    elif omic1 == 'RNA' and omic2 == 'ATAC':
        trainFactorsFile = 'src/MOFA2/mofa_atac_factors_%s%s_training.tsv' % (omic1, omic2)

        data_path1 = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/%s.npy' % omic1.lower()
        data_path2 = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/%s.npy' % omic2.lower()


        train_ind_file = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/trainInd.npy'
        val_ind_file = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/validInd.npy'
        test_ind_file = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/testInd.npy'

        sample_namesFile = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/sampleNames.npy'

        weightsFile1 = 'src/MOFA2/mofa_atac_weights_%s%s_%s.tsv' % (omic1, omic2, omic1)
        weightsFile2 = 'src/MOFA2/mofa_atac_weights_%s%s_%s.tsv' % (omic1, omic2, omic2)



    else:
        train_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/trainInd.npy'
        val_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/validInd.npy'
        test_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/testInd.npy'

        sample_namesFile = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD_sampleNames.npy'

        weightsFile1 = 'src/MOFA2/mofa_cite_weights_rna.tsv'
        weightsFile2 = 'src/MOFA2/mofa_cite_weights_adt.tsv'

    likelihoods = [sys.argv[4], sys.argv[5]]
    n_categories = [int(sys.argv[6]), int(sys.argv[7])]


    dirName = 'embeddings/results/test_%s_%s_mofa_%s_%s/MOFA+/' % (omic1, omic2, omic1, omic2)

except:
    trainFactorsFile = 'src/MOFA2/mofa_factors_training.tsv'

    data_path1 = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/RNA.npy'
    data_path2 = '/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/ADT.npy'

    train_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/trainInd.npy'
    val_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/validInd.npy'
    test_ind_file = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/testInd.npy'

    sample_namesFile = '/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD_sampleNames.npy'

    weightsFile1 = 'src/MOFA2/mofa_cite_weights_rna.tsv'
    weightsFile2 = 'src/MOFA2/mofa_cite_weights_adt.tsv'

    likelihoods = ['nb', 'normal']
    n_categories = [-1, -1]

    dirName = 'embeddings/results/test_RNA_ADT_mofa_RNA_ADT/MOFA/'

ztrainDF = pd.read_csv(trainFactorsFile, index_col=0, sep='\t')
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

logger.output_file = 'src/MOFA2/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info("Selected device: {}".format(device))
torch.manual_seed(1)

try:
    save_dir = 'src/MOFA2/imputation' + omic2 + 'from' + omic1
except:
    save_dir = 'src/MOFA2/imputation2from1'
os.makedirs(save_dir, exist_ok=True)
ckpt_dir = save_dir + '/checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
logs_dir = save_dir + '/logs'



logger.info('Likelihood for omic1: %s' % likelihoods[0])
logger.info('Likelihood for omic2: %s' % likelihoods[1])


# Initialize GLMs

if likelihoods[1] not in set(['nb', 'nbm', 'zinb']):
    net2from1 = OmicRegressor(ztrain.shape[1], X2train.shape[1], distribution=likelihoods[1], optimizer_name='Adam', lr=0.0001, n_categories=n_categories[1])
else:
    net2from1 = OmicRegressorSCVI(ztrain.shape[1], X2train.shape[1], distribution=likelihoods[1], optimizer_name='Adam', lr=0.0001, use_batch_norm=False, log_input=False)

net2from1 = net2from1.to(device).double()

if likelihoods[0] not in set(['nb', 'nbm', 'zinb']):
    net1from2 = OmicRegressor(ztrain.shape[1], X1train.shape[1], distribution=likelihoods[0], optimizer_name='Adam', lr=0.0001, n_categories=n_categories[0])
else:
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

metricsTestIndividual2from1 = evaluatePerDatapoint(net2from1, device, test_loader_individual, True)


logger.info('Validation performance, imputation error modality 2 from 1')
for m in metricsValidation:
    logger.info('%s\t%.4f' % (m, metricsValidation[m]))

logger.info('Test performance, imputation error modality 2 from 1')
for m in metricsTest:
    logger.info('%s\t%.4f' % (m, metricsTest[m]))


# modality 1 from 2

try:
    save_dir = 'src/MOFA2/imputation' + omic1 + 'from' + omic2
except:
    save_dir = 'src/MOFA2/imputation1from2'
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
try:
    checkpoint = torch.load(ckpt_dir + '/model_best.pth.tar')
except FileNotFoundError:
    logger.info('Training did not complete properly, using last epoch')
    checkpoint = torch.load(ckpt_dir + '/model_last.pth.tar')


net1from2.load_state_dict(checkpoint['state_dict'])


metricsValidation = evaluateUsingBatches(net1from2, device, valid_loader, True)
metricsTest = evaluateUsingBatches(net1from2, device, test_loader, True)

logger.info('Validation performance, imputation error modality 1 from 2')
for m in metricsValidation:
    logger.info('%s\t%.4f' % (m, metricsValidation[m]))

logger.info('Test performance, imputation error modality 1 from 2')
for m in metricsTest:
    logger.info('%s\t%.4f' % (m, metricsTest[m]))

metricsTestIndividual1from2 = evaluatePerDatapoint(net1from2, device, test_loader_individual, True)



logger.info('\n\n')

weights1 = pd.read_csv(weightsFile1, index_col=0, sep='\t')
weights2 = pd.read_csv(weightsFile2, index_col=0, sep='\t')

zdim = weights1.shape[1]
np.random.seed(1)
z = np.random.multivariate_normal(np.zeros(zdim), np.eye(zdim), 2000)

fakeRNA = np.array(weights1).dot(z.T).T
fakeADT = np.array(weights2).dot(z.T).T

try:
    coh = evaluate_generation(torch.tensor(fakeRNA), torch.tensor(fakeADT), omic1, omic2)
except:
    coh = evaluate_generation(torch.tensor(fakeRNA), torch.tensor(fakeADT), 'RNA', 'ADT')

logger.info('Coherence: %.4f' % coh)

logger.info('\n\n')

try:
    individualPerfomanceSave = 'src/MOFA2/imputation_' + omic1 + omic2 + '_performancePerDatapont.pkl'
except:
    individualPerfomanceSave = 'src/MOFA2/imputation_RNAADT_performancePerDatapont.pkl'

with open(individualPerfomanceSave, 'wb') as f:
    pickle.dump({'1from2': metricsTestIndividual1from2, '2from1': metricsTestIndividual2from1}, f)


if task > 1:

    if 'omic2' in locals() and omic2 == 'ATAC':
        classLabels = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/celltype_l2.npy')
        labelNames = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scatac/celltypes_l2.npy', allow_pickle=True)
    else:
        classLabels = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_cellType_l3.npy')
        labelNames = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_cellTypes_l3.npy', allow_pickle=True)

    ytrain = classLabels[trainInd]
    yvalid = classLabels[validInd]
    ytest = classLabels[testInd]

    logger.info('Test performance, classification task, linear classifier, modality 1')
    _, acc, pr, rc, f1, mcc, confMat, CIs = classification(embDict['ztrain'][0], ytrain, embDict['zvalidation'][0], yvalid, embDict['ztest'][0], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
    performance1 = [acc, pr, rc, f1, mcc, confMat]
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


    pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

    logger.info('Test performance, classification task, linear classifier, modality 2')
    _, acc, pr, rc, f1, mcc, confMat, CIs = classification(embDict['ztrain'][1], ytrain, embDict['zvalidation'][1], yvalid, embDict['ztest'][1], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
    performance2 = [acc, pr, rc, f1, mcc, confMat]
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

    pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

    logger.info('Test performance, classification task, linear classifier, both modalities')
    _, acc, pr, rc, f1, mcc, confMat, CIs = classification(embDict['ztrain'][2], ytrain, embDict['zvalidation'][2], yvalid, embDict['ztest'][2], ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
    performance12 = [acc, pr, rc, f1, mcc, confMat]
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

    pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}

    if 'omic2' in locals() and omic2 == 'ATAC':
        level = 'l2'
    else:
        level = 'l3'
        omic1 = 'RNA'
        omic2 = 'ADT'

    # -----------------------------------------------------------------
    logger.info('Test performance, classification task, non-linear classifier, modality 1')
    _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(embDict['ztrain'][0], ytrain, embDict['zvalidation'][0], yvalid, embDict['ztest'][0], ytest, 'type-classifier/eval/' + level + '/mofa_' + omic1 + '/')
    performance1 = [acc, pr, rc, f1, mcc, confMat]
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


    mlp_pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

    logger.info('Test performance, classification task, non-linear classifier, modality 2')
    _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(embDict['ztrain'][1], ytrain, embDict['zvalidation'][1], yvalid, embDict['ztest'][1], ytest, 'type-classifier/eval/' + level + '/mofa_' + omic2 + '/')
    performance2 = [acc, pr, rc, f1, mcc, confMat]
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

    mlp_pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

    logger.info('Test performance, classification task, non-linear classifier, both modalities')
    _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(embDict['ztrain'][2], ytrain, embDict['zvalidation'][2], yvalid, embDict['ztest'][2], ytest, 'type-classifier/eval/' + level + '/mofa_' + omic1 + '_' + omic2 + '/')
    performance12 = [acc, pr, rc, f1, mcc, confMat]
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

    mlp_pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}

    save_dir = 'src/MOFA2/'
    if 'omic2' in locals() and omic2 == 'ATAC':
        ext = 'atac_'
    else:
        ext = ''

    logger.info("Saving results")


    with open(save_dir + "/" + ext + "MOFA_task2_results.pkl", 'wb') as f:
        pickle.dump({'omic1': pr1, 'omic2': pr2, 'omic1+2': pr12, 'omic1-mlp': mlp_pr1, 'omic2-mlp': mlp_pr2, 'omic1+2-mlp': mlp_pr12}, f)
