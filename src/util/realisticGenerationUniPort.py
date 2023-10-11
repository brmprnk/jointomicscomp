import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from src.util.evaluate import *
import src.util.logger as logger
from src.nets import MLP
from torch.utils.data import DataLoader
from src.util.trainTypeClassifier import CustomDataset
import sys
import uniport as up
import anndata as ada
import numpy as np
from scipy.sparse import csr_matrix
from uniport.model.vae import VAE
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

logger.output_file = './'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info("Selected device: {}".format(device))
torch.manual_seed(1)


n_modalities = 2

# Load in data
unscaledomics = [np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/RNA.npy'), np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/ADT.npy')]
modalities = ['RNA', 'ADT']

omics = []
scalers = []
for i, omic1 in enumerate(unscaledomics):
    m = np.min(omic1)
    if m < 0:
        ss = MinMaxScaler().fit(omic1)
    else:
        ss = MaxAbsScaler().fit(omic1)

    omics.append(ss.transform(omic1))
    scalers.append(ss)

labels = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltype_l3.npy')
labeltypes = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltypes_l3.npy', allow_pickle=True)



# Use predefined split
train_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/trainInd.npy')
val_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/validInd.npy')
test_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/testInd.npy')

args = dict()
args['latent_dim'] = '256-128-32'
args['pre_trained'] = 'results/uniport/results/cite-uniport-6_RNA_ADT/UniPort/checkpoint/'
args['data1'] = 'RNA'
args['data2'] = 'ADT'
args['batch_size'] = 64



omics_train = [omic[train_ind] for omic in omics]
adtrain = []
for i, omic in enumerate(omics_train):
    ds = ada.AnnData(csr_matrix(omic))
    ds.obs['source'] = args['data%d' % (i+1)]
    ds.obs['domain_id'] = 0
    ds.obs['domain_id'] = ds.obs['domain_id'].astype('category')
    adtrain.append(ds)


omics_val = [omic[val_ind] for omic in omics]
omics_test = [omic[test_ind] for omic in omics]

ytrain = labels[train_ind]
yvalid = labels[val_ind]
ytest = labels[test_ind]

unscaledomicslog = [np.log(1 + omic) for omic in unscaledomics]

unscaledomicslog_train = [omic[train_ind] for omic in unscaledomicslog]
unscaledomicslog_val = [omic[val_ind] for omic in unscaledomicslog]
unscaledomicslog_test = [omic[test_ind] for omic in unscaledomicslog]


# Number of features
input_dims = [5000, 217]

likelihoods = ['nb', 'nbm']

# use real training data to do PCA for each modality
pca1 = PCA(n_components=32, whiten=False, svd_solver='full')
pca1.fit(unscaledomicslog_train[0])

X1train = pca1.transform(unscaledomicslog_train[0])
X1valid = pca1.transform(unscaledomicslog_val[0])
X1test = pca1.transform(unscaledomicslog_test[0])

pca2 = PCA(n_components=32, whiten=False, svd_solver='full')
pca2.fit(unscaledomicslog_train[1])

X2train = pca2.transform(unscaledomicslog_train[1])
X2valid = pca2.transform(unscaledomicslog_val[1])
X2test = pca2.transform(unscaledomicslog_test[1])


Xtrain = [X1train, X2train]
Xvalid = [X1valid, X2valid]
Xtest = [X1test, X2test]
pcas = [pca1, pca2]

layers = args['latent_dim'].split('-')

enc_arch = []
for i in range(len(layers)-1):
    enc_arch.append(['fc', int(layers[i]), 1, 'relu'])

enc_arch.append(['fc', int(layers[-1]), '', ''])

state = torch.load(args['pre_trained'] + 'config.pt')
enc, dec, n_domain, ref_id, num_gene = state['enc'], state['dec'], state['n_domain'], state['ref_id'], state['num_gene']
model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain, mode='v')
model.load_model(args['pre_trained'] + 'model.pt')
model.to(device)
model.eval()


logger.success("Loaded trained UniPort model.")


# embed test data
adtest = []
for i, omic in enumerate(omics_test):
    ds = ada.AnnData(csr_matrix(omic))
    ds.obs['source'] = args['data%d' % (i+1)]
    ds.obs['domain_id'] = 0
    ds.obs['domain_id'] = ds.obs['domain_id'].astype('category')
    adtest.append(ds)

with torch.no_grad():
    ad = up.Run(adatas=adtest, mode='v', out='project', batch_size=args['batch_size'], gpu=0, outdir=args['pre_trained']+'../', seed=124, batch_key='domain_id', source_name='source')
    ztest = ad.obsm['project']


imputedADT = model.decoder(torch.tensor(ztest, device=device), 1).detach().cpu().numpy()
imputedADT = scalers[1].inverse_transform(imputedADT)

imputedADT = np.log(1 + imputedADT)

imputed2test = pca2.transform(imputedADT)


# logger.info('defining model with %d classes' % np.unique(y).shape[0])
clfSingle = MLP(Xtrain[1].shape[1], 64, np.unique(ytrain).shape[0])
clfSingle = clfSingle.double()

clfSingle = clfSingle.to(device)

# sth went wrong with the level when saving, this l2 should be correct
# so actually l3 is used, if it were wrong code with crash because the
# nubmer of labels in the test data are 57 (ie level 3)
checkpoint = torch.load('type-classifier/eval/l2/baseline_ADT/checkpoint/model_best.pth.tar')
clfSingle.load_state_dict(checkpoint['state_dict'])
clfSingle.eval()

# real dataset, one modality
testDatasetReal = CustomDataset(Xtest[1], ytest)
test_loader_mlpReal = DataLoader(testDatasetReal, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

# imputed dataset, one modality
testDatasetImputed = CustomDataset(imputed2test, ytest)
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

logger.info('Test performance, ADT, real data')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
performanceSingleReal = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(ytest, ypred_testImputed)

logger.info('Test performance, ADT, imputed data')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
performanceSingleImputed = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

agreement = np.mean(ypred_testReal == ypred_testImputed)
logger.info('Prediction agreement between real and predicted data: %.4f' % agreement)

# do the same for classifier using both modalities
# ---------------------------------------------------------------------------------------------------------- #

clfDouble = MLP(Xtrain[1].shape[1] * 2, 64, np.unique(ytrain).shape[0])
clfDouble = clfDouble.double()

clfDouble = clfDouble.to(device)

checkpoint = torch.load('type-classifier/eval/l2/baseline_RNA_ADT/checkpoint/model_best.pth.tar')
clfDouble.load_state_dict(checkpoint['state_dict'])
clfDouble.eval()

# real dataset, one modality
testDatasetReal = CustomDataset(np.hstack(Xtest), ytest)
test_loader_mlpReal = DataLoader(testDatasetReal, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

# imputed dataset, one modality

testDatasetImputed = CustomDataset(np.hstack((Xtest[0], imputed2test)), ytest)

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

logger.info('Test performance, ADT, real data')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
performanceSingleReal = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(ytest, ypred_testImputed)

logger.info('Test performance, ADT, imputed data')
logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
performanceSingleImputed = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

agreement = np.mean(ypred_testReal == ypred_testImputed)
logger.info('Prediction agreement between real and predicted data: %.4f' % agreement)
