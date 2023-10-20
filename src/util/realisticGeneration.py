import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge, SGDClassifier
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
omics = [np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/RNA.npy'), np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/ADT.npy')]
modalities = ['RNA', 'ADT']

labels = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltype_l3.npy')
labeltypes = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltypes_l3.npy', allow_pickle=True)



# Use predefined split
train_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/trainInd.npy')
val_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/validInd.npy')
test_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/testInd.npy')


omics = [np.log(1 + omic) for omic in omics]

omics_train = [omic[train_ind] for omic in omics]
omics_val = [omic[val_ind] for omic in omics]
omics_test = [omic[test_ind] for omic in omics]

ytrain = labels[train_ind]
yvalid = labels[val_ind]
ytest = labels[test_ind]

# Number of features
input_dims = [5000, 217]

likelihoods = ['nb', 'nbm']

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


encoder_layers = [int(kk) for kk in '256-256-32'.split('-')]
decoder_layers = encoder_layers[::-1][1:]

# load pre-trained MoE model
model = MixtureOfExperts(input_dims, encoder_layers, decoder_layers,
 likelihoods, False, 0.0, 'Adam', 0.0001, 0.0001, 'laplace', 1.0, 10, [1., 1.], [-1, -1], [True, True])


model.double()

if device == torch.device('cuda'):
    model.cuda()

checkpoint = torch.load('results/tmpnbcite/results/likelihood-cite-moe-6_RNA_ADT/MoE/checkpoint/model_best.pth.tar')

for i in range(n_modalities):
    #print(i)
    model.encoders[i].load_state_dict(checkpoint['state_dict_enc'][i])
    model.decoders[i].load_state_dict(checkpoint['state_dict_dec'][i])

logger.success("Loaded trained MixtureOfExperts model.")

# embed test data
dataTest = [torch.tensor(omic1, device=device) for omic1 in omics_test]
datasetTest = MultiOmicsDataset(dataTest)

test_loader = torch.utils.data.DataLoader(datasetTest, batch_size=64, shuffle=False, num_workers=0,
                          drop_last=False)


imputedRNA = np.zeros(omics_test[0].shape)
imputedADT = np.zeros(omics_test[1].shape)

ind = 0
for data in test_loader:
    batch = (data[0][0].double(), data[1][0].double())

    _, x_tmp = model.embedAndReconstruct(batch)

    imputedRNA[ind:ind + 64] = x_tmp[1][0].mean.cpu().detach().numpy()
    imputedADT[ind:ind + 64] = x_tmp[0][1].mean.cpu().detach().numpy()

    ind += 64

imputedRNA = np.log(1 + imputedRNA)
imputedADT = np.log(1 + imputedADT)

imputed1test = pca1.transform(imputedRNA)
imputed2test = pca2.transform(imputedADT)
imputedTest = [imputed1test, imputed2test]


performances = []
for i, modality in enumerate(modalities):

    # logger.info('defining model with %d classes' % np.unique(y).shape[0])
    clfSingle = MLP(Xtrain[i].shape[1], 64, np.unique(ytrain).shape[0])
    clfSingle = clfSingle.double()

    clfSingle = clfSingle.to(device)

    # sth went wrong with the level when saving, this l2 should be correct
    # so actually l3 is used, if it were wrong code with crash because the
    # nubmer of labels in the test data are 57 (ie level 3)
    checkpoint = torch.load('type-classifier/eval/l2/baseline_%s/checkpoint/model_best.pth.tar' % modality)
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

    checkpoint = torch.load('type-classifier/eval/l2/baseline_RNA_ADT/checkpoint/model_best.pth.tar')
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
