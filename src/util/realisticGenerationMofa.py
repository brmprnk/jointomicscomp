import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from src.util.evaluate import *
import src.util.logger as logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter
from src.nets import MLP
from src.CGAE.model import evaluateUsingBatches, MultiOmicsDataset
from torch.utils.data import DataLoader
from src.util.trainTypeClassifier import CustomDataset
from src.MoE.model import MixtureOfExperts
import sys


logger.output_file = './'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info("Selected device: {}".format(device))
torch.manual_seed(1)


n_modalities = 2

# Load in data
omics = [np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/RNA.npy'), np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/ADT.npy')]
modalities = ['RNA', 'ADT']

labels = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_cellType_l3.npy')
labeltypes = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_ADT_5000MAD_cellTypes_l3.npy', allow_pickle=True)



# Use predefined split
train_ind = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/trainInd.npy')
val_ind = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/validInd.npy')
test_ind = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/task1/testInd.npy')

omics_train = [omic[train_ind] for omic in omics]
omics_val = [omic[val_ind] for omic in omics]
omics_test = [omic[test_ind] for omic in omics]

ytrain = labels[train_ind]
yvalid = labels[val_ind]
ytest = labels[test_ind]


# use real training data to do PCA for each modality
pca1 = PCA(n_components=32, whiten=False, svd_solver='full')
pca1.fit(omics_train[0])

X1train = pca1.transform(omics_train[0])
X1valid = pca1.transform(omics_val[0])
X1test = pca1.transform(omics_test[0])

pca2 = PCA(n_components=32, whiten=False, svd_solver='full')
pca2.fit(omics_train[1])

X2train = pca2.transform(omics_train[1])
X2valid = pca2.transform(omics_val[1])
X2test = pca2.transform(omics_test[1])


Xtrain = [X1train, X2train]
Xvalid = [X1valid, X2valid]
Xtest = [X1test, X2test]
pcas = [pca1, pca2]


# load pre-trained MoFA model and learn projection
trainFactorsFile = 'src/MOFA2/mofa_cite_factors_training.tsv'

ztrainDF = pd.read_csv(trainFactorsFile, index_col=0, sep='\t')
ztrain = np.array(ztrainDF)

mapper1 = LinearRegression()
mapper1.fit(omics_train[0], ztrain)
print('Projection from modality 1 to z\nMean R^2 on training data: %.4f' % mapper1.score(omics_train[0], ztrain))


mapper2 = LinearRegression()
mapper2.fit(omics_train[1], ztrain)
print('Projection from modality 2 to z\nMean R^2 on training data: %.4f' % mapper2.score(omics_train[1], ztrain))


weightsFile1 = 'src/MOFA2/mofa_cite_weights_rna.tsv'
weightsFile2 = 'src/MOFA2/mofa_cite_weights_adt.tsv'

weights1 = pd.read_csv(weightsFile1, index_col=0, sep='\t')
weights2 = pd.read_csv(weightsFile2, index_col=0, sep='\t')




# embed test data
imputedRNA = np.array(weights1).dot(mapper2.predict(omics_test[1]).T).T

imputedADT = np.array(weights2).dot(mapper1.predict(omics_test[0]).T).T


imputed1test = pca1.transform(imputedRNA)
imputed2test = pca2.transform(imputedADT)
imputedTest = [imputed1test, imputed2test]


performances = []
for i, modality in enumerate(modalities):

    # logger.info('defining model with %d classes' % np.unique(y).shape[0])
    clfSingle = MLP(Xtrain[i].shape[1], 64, np.unique(ytrain).shape[0])
    clfSingle = clfSingle.double()

    clfSingle = clfSingle.to(device)

    checkpoint = torch.load('type-classifier/eval/l3/baseline_%s/checkpoint/model_best.pth.tar' % modality)
    clfSingle.load_state_dict(checkpoint['state_dict'])
    clfSingle.eval()

    # real dataset, one modality
    testDatasetReal = CustomDataset(Xtest[i], ytest)
    test_loader_mlpReal = DataLoader(testDatasetReal, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    # imputed dataset, one modality
    testDatasetImputed = CustomDataset(imputedTest[i], ytest)
    test_loader_mlpImputed = DataLoader(testDatasetImputed, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    ypred_testReal = np.zeros(ytest.shape, int)
    ypred_testImputed = np.zeros(ytest.shape, int)

    with torch.no_grad():
        ind = 0
        b = 64
        for x, y in test_loader_mlpReal:
            y_pred = clfSingle.forward(x[0].double().to(device))

            ypred_testReal[ind:ind+b] = torch.argmax(y_pred, dim=1).cpu().detach().numpy()

            ind += b

        ind = 0

        for x, y in test_loader_mlpImputed:
            y_pred = clfSingle.forward(x[0].double().to(device))

            ypred_testImputed[ind:ind+b] = torch.argmax(y_pred, dim=1).cpu().detach().numpy()

            ind += b



    # evaluate
    acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(ytest, ypred_testReal)

    logger.info('Test performance, %s, real data' % modality)
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
    performanceSingleReal = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

    acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(ytest, ypred_testImputed)

    logger.info('Test performance, %s, imputed data' % modality)
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
    performanceSingleImputed = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

    agreement = np.mean(ypred_testReal == ypred_testImputed)
    logger.info('Prediction agreement between real and predicted data: %.4f' % agreement)

    # do the same for classifier using both modalities
    # ---------------------------------------------------------------------------------------------------------- #

    clfDouble = MLP(Xtrain[i].shape[1] * 2, 64, np.unique(ytrain).shape[0])
    clfDouble = clfDouble.double()

    clfDouble = clfDouble.to(device)

    checkpoint = torch.load('type-classifier/eval/l3/baseline_RNA_ADT/checkpoint/model_best.pth.tar')
    clfDouble.load_state_dict(checkpoint['state_dict'])
    clfDouble.eval()

    # real dataset, one modality
    testDatasetReal = CustomDataset(np.hstack(Xtest), ytest)
    test_loader_mlpReal = DataLoader(testDatasetReal, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    # imputed dataset, one modality

    if i == 0:
        testDatasetImputed = CustomDataset(np.hstack((imputedTest[0], Xtest[1])), ytest)
    else:
        testDatasetImputed = CustomDataset(np.hstack((Xtest[0], imputedTest[1])), ytest)

    test_loader_mlpImputed = DataLoader(testDatasetImputed, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    ypred_testReal = np.zeros(ytest.shape, int)
    ypred_testImputed = np.zeros(ytest.shape, int)

    with torch.no_grad():
        ind = 0
        b = 64
        for x, y in test_loader_mlpReal:
            y_pred = clfDouble.forward(x[0].double().to(device))

            ypred_testReal[ind:ind+b] = torch.argmax(y_pred, dim=1).cpu().detach().numpy()

            ind += b

        ind = 0

        for x, y in test_loader_mlpImputed:
            y_pred = clfDouble.forward(x[0].double().to(device))

            ypred_testImputed[ind:ind+b] = torch.argmax(y_pred, dim=1).cpu().detach().numpy()

            ind += b



    # evaluate
    acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(ytest, ypred_testReal)

    logger.info('Test performance, %s, real data' % modality)
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
    performanceSingleReal = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

    acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(ytest, ypred_testImputed)

    logger.info('Test performance, %s, imputed data' % modality)
    logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
    performanceSingleImputed = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

    agreement = np.mean(ypred_testReal == ypred_testImputed)
    logger.info('Prediction agreement between real and predicted data: %.4f' % agreement)
