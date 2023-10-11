from src.totalVI.preprocessCITE import loadCITE, maskRNA, maskProteins
import torch
from src.util import logger
import os
from scvi.model import TOTALVI
import numpy as np
import pickle
from torch.distributions import Normal, Independent, Bernoulli
from torch.utils.data import DataLoader
from src.baseline.baseline import classification, classificationMLP
from src.util.evaluate import evaluate_imputation, evaluate_classification, evaluate_generation
import src.util.mydistributions as mydist
import src.util.mydistributions as mydist
from sklearn.decomposition import PCA
from src.nets import MLP
from src.util.trainTypeClassifier import CustomDataset
from src.CGAE.model import MultiOmicsDataset
# Load in data
omics = [np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/RNA.npy'), np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/ADT.npy')]
modalities = ['RNA', 'ADT']

labels = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltype_l3.npy')
labeltypes = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/celltypes_l3.npy', allow_pickle=True)



# Use predefined split
train_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/trainInd.npy')
val_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/validInd.npy')
test_ind = np.load('/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/data/scvi-cite/testInd.npy')

counts = [omics[0], omics[1]]

counts_train = [omic[train_ind] for omic in counts]
counts_val = [omic[val_ind] for omic in counts]
counts_test = [omic[test_ind] for omic in counts]


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
args['zdim'] = 32
args['mudata_path'] = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/'
args['enc_distribution'] = 'normal'
args['n_neurons_per_layer'] = 256
args['n_layers'] = 2
args['nomics'] = 2
args['pre_trained'] = 'results/tmpnbcite/results/likelihood-cite-totalvi-test-63_RNA_ADT/totalVI/checkpoint/'

logger.output_file = './'
device = torch.device('cuda')
logger.info("Selected device: {}".format(device))
torch.manual_seed(1)

# save_dir = os.path.join(args['save_dir'], '{}'.format('totalVI'))
# os.makedirs(save_dir)
# ckpt_dir = save_dir + '/checkpoint'


n_modalities = args['nomics']
assert n_modalities == 2

# Load in data
mudataTrain, mudataTest = loadCITE(save=False, dataPrefix=args['mudata_path'])

params = dict(n_latent=args['zdim'], latent_distribution=args['enc_distribution'], n_hidden=args['n_neurons_per_layer'], n_layers_encoder=args['n_layers'], n_layers_decoder=args['n_layers'])

net = TOTALVI.load(args['pre_trained'], adata=mudataTest, accelerator='gpu', device='auto')

mudataTestRnaOnly = maskProteins(mudataTest)

# with torch.no_grad():
#     llstest = net.get_latent_library_size(mudataTest, give_mean=False)
#     z1test = net.get_latent_representation(mudataTestRnaOnly, give_mean=True, return_dist=False)
#
#     pdict = net.module.generative(torch.tensor(z1test).to(device), torch.tensor(llstest).to(device), torch.zeros(z1test.shape[0]), torch.zeros(z1test.shape[0]))
#     pdictProt = pdict['py_']
#
#     pred = mydist.NegativeBinomialMixture(pdictProt['rate_back'], pdictProt['rate_fore'], pdictProt['r'], pdictProt['mixing'])

########################################################
post = net._make_data_loader(adata=mudataTestRnaOnly, shuffle=False, batch_size=64)

gene_mask = slice(None)
protein_mask = slice(None)
scale_list_gene = []
scale_list_pro = []

ll = []
imputedADT = np.zeros((Xtest[1].shape[0], omics[1].shape[1]))


from scvi import REGISTRY_KEYS
n_samples = 1
transform_batch = [0]
library_size = 'latent'
ind = 0
with torch.no_grad():
    for tensors in post:
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        px_scale = torch.zeros_like(x)[..., gene_mask]
        py_scale = torch.zeros_like(y)[..., protein_mask]
        if n_samples > 1:
            px_scale = torch.stack(n_samples * [px_scale])
            py_scale = torch.stack(n_samples * [py_scale])
        for b in transform_batch:
            generative_kwargs = {"transform_batch": b}
            inference_kwargs = {"n_samples": n_samples}
            _, generative_outputs = net.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                generative_kwargs=generative_kwargs,
                compute_loss=False,
            )
            if library_size == "latent":
                px_scale += generative_outputs["px_"]["rate"].cpu()[..., gene_mask]
            else:
                px_scale += generative_outputs["px_"]["scale"].cpu()[..., gene_mask]

            py_ = generative_outputs["py_"]

            pred = mydist.NegativeBinomialMixture(py_['rate_back'], py_['rate_fore'], py_['r'], py_['mixing'])
            # ll = ll + llbatch
            ll.append(pred)

            imputedADT[ind:ind+64] = pred.mean.detach().cpu().numpy()
            ind += 64



imputedADT = net.get_normalized_expression(mudataTestRnaOnly, return_numpy=True)[1]
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


counts_test2 = [torch.tensor(x) for x in counts_test]

datasetTest = MultiOmicsDataset(counts_test2)
test_loader = DataLoader(datasetTest, batch_size=64, shuffle=False, num_workers=0,drop_last=False)

llvals = []
for i, data in enumerate(test_loader):
    #batch = (data[0][0].double(), data[1][0].double())
    adt = data[1][0].double().to(device)

    v = torch.sum(ll[i].log_prob(adt),1).detach().cpu().numpy()
    llvals += list(v)
