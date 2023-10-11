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
from src.PoE.model import PoE
import src.PoE.datasets as datasets
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



args = dict()
args['cuda'] = True
args['latent_dim'] = '256-64'
args['num_features1'] = 5000
args['num_features2'] = 217
args['dropout_probability'] = 0.0
args['use_batch_norm'] = False
args['likelihood1'] = 'nb'
args['likelihood2'] = 'nbm'

args['data1'] = 'RNA'
args['data2'] = 'ADT'
args['log_inputs'] = 'True'

model = PoE(args)

model.double()

if device == torch.device('cuda'):
    model.cuda()

checkpoint = torch.load('results/tmpnbcite/results/likelihood-cite-poe-15_RNA_ADT/PoE/checkpoint/model_best.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
logger.success("Loaded trained ProductOfExperts model.")

######### !!!!!!! here #######

args['data_path1'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/RNA.npy'
args['data_path2'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/ADT.npy'
args['labels'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltype_l3.npy'
args['labelnames'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltypes_l3.npy'

args['train_ind'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/trainInd.npy'
args['val_ind'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/validInd.npy'
args['test_ind'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/testInd.npy'
args['batch_size'] = 64

save_dir = 'results/tmpnbcite/results/likelihood-cite-poe-15_RNA_ADT/PoE/'

# embed test data
tcga_data = datasets.TCGAData(args, save_dir=save_dir)
test_dataset = tcga_data.get_data_partition("test")

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False)  #


imputedRNA = np.zeros(omics_test[0].shape)
imputedADT = np.zeros(omics_test[1].shape)

model.eval()
ind = 0
with torch.no_grad():
    for batch_idx, (omic1, omic2) in enumerate(test_loader):

        if args['cuda']:
            omic1 = omic1.cuda()
            omic2 = omic2.cuda()

        # compute reconstructions using each of the individual modalities
        (_, omic1_recon_omic2, _, _) = model.forward(omic1=omic1)

        (omic2_recon_omic1, _, _, _) = model.forward(omic2=omic2)

        imputedRNA[ind:ind+args['batch_size']] = omic2_recon_omic1.mean.cpu().detach().numpy()
        imputedADT[ind:ind+args['batch_size']] = omic1_recon_omic2.mean.cpu().detach().numpy()

        ind += args['batch_size']

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
