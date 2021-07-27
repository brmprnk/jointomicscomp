import argparse
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from mord import OrdinalRidge
from evaluate import *


def impute(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, alphas, criterion='mse'):
    # returns imputed values as well as evaluation of the imputation based on mse and rsquared

    if criterion == 'mse':
        ind = 0
    else:
        assert criterion == 'rsquared'
        ind = 1

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = Ridge(alpha=a, fit_intercept=True, normalize=False, random_state=1)

        # train
        model.fit(Xtrain, Ytrain)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_imputation(Yvalid, model.predict(Xvalid))[ind]

    if criterion == 'mse':
        bestModel = models[np.argmin(validationPerformance)]
    else:
        bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(Xtest)

    return predictions, evaluate_imputation(Ytest, predictions)


def ordinalRegression(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, alphas, criterion='mcc'):
    # returns the predicted stage, as well as accuracy measures

    if criterion == 'acc':
        ind = 0
    elif criterion == 'pr':
        ind = 1
    elif criterion == 'rc':
        ind = 2
    elif criterion == 'f1':
        ind = 3
    else:
        assert criterion == 'mcc'
        ind = 4

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = OrdinalRidge(alpha=a, fit_intercept=True, normalize=False, random_state=1)

        # train
        model.fit(Xtrain, Ytrain)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_classification(Yvalid, model.predict(Xvalid))[ind]

    bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(Xtest).astype(int)

    return predictions, evaluate_classification(Ytest, predictions)


def classification(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, alphas, criterion='mcc'):
    # returns the predicted stage, as well as accuracy measures

    if criterion == 'acc':
        ind = 0
    elif criterion == 'pr':
        ind = 1
    elif criterion == 'rc':
        ind = 2
    elif criterion == 'f1':
        ind = 3
    else:
        assert criterion == 'mcc'
        ind = 4

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = LinearSVC(penalty='l2', loss='hinge', C=a, multi_class='ovr', fit_intercept=True, random_state=1)

        # train
        model.fit(Xtrain, Ytrain)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_classification(Yvalid, model.predict(Xvalid))[ind]

    bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(Xtest).astype(int)

    return predictions, evaluate_classification(Ytest, predictions)




parser = argparse.ArgumentParser(description='')
# Learning hyperparameters
parser.add_argument('--train-data',     dest='XtrainFile',   type=str)
parser.add_argument('--train-labels',   dest='YtrainFile',   type=str,       default=None)
parser.add_argument('--valid-data',     dest='XvalidFile',   type=str)
parser.add_argument('--valid-labels',   dest='YvalidFile',   type=str,       default=None)
parser.add_argument('--test-data',      dest='XtestFile',    type=str)
parser.add_argument('--test-labels',    dest='YtestFile',    type=str,       default=None)
# whether to `impute` one datasource from the other, `classify` (cell type), or `rank` (stage prediction)
parser.add_argument('--task',           dest='task',         type=str,       default='impute')
parser.add_argument('--results-file',   dest='resultsFile',  type=str,       default='./res.pkl')
parser.add_argument('--baseline',       dest='baseline',     type=bool,      default=False)

args = parser.parse_args()

alphas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10., 20.])
Cs = 1. / alphas


# load features (raw GE/ME/etc or VAE embeddings) from K data sources, N tumors, F features
# a pickle file with a list of length K, each element containing a N x F npy array
with open(args.XtrainFile, 'rb') as f:
    XtrainList = pickle.load(f)

with open(args.XvalidFile, 'rb') as f:
    XvalidList = pickle.load(f)

with open(args.XtestFile, 'rb') as f:
    XtestList = pickle.load(f)


if args.task == 'classify':
    # a pickle file with the class label npy array with shape (N,)
    with open(args.YtrainFile, 'rb') as f:
        Ytrain = pickle.load(f)

    with open(args.XvalidFile, 'rb') as f:
        Yvalid = pickle.load(f)

    with open(args.XtestFile, 'rb') as f:
        Ytest = pickle.load(f)


    Nclasses = np.unique(Ytrain).shape[0]

    # MOFA, MoE, PoE, MVIB learn a common z --> X will be just an array N x F
    # CGAE has  a separate z for each modality --> so evaluate K times
    # baseline has the raw data and requires pca --> in both of these cases there will be a list of length K, with N x F matrices as elements

    if type(XtrainList) == list:
        if args.baseline:
            pcTrainList = []
            pcValidList = []
            pcTestList = []

            for i in range(XtrainList):
                pca = PCA(n_components=50, whiten=False, svd_solver='full')
                pca.fit(XtrainList[i])

                pcTrainList.append(pca.transform(XtrainList[i]))
                pcValidList.append(pca.transform(XvalidList[i]))
                pcTestList.append(pca.transform(XtestList[i]))

            Xtrain = np.hstack(pcTrainList)
            Xvalid = np.hstack(pcValidList)
            Xtest = np.hstack(pcTestList)

            assert Xtrain.shape[0] == Xtest.shape[0]
            assert Xtrain.shape[1] == 50 * len(XtrainList)

            _, acc, pr, rc, f1, mcc, confMat = classification(XtrainList, Ytrain, XvalidList, Yvalid, XtestList, Ytest, Cs, 'mcc')

        else:
            K = len(XtrainList)
            assert len(XvalidList) == K
            assert len(XtestList) == K

            acc = np.zeros(K)
            pr = np.zeros(K)
            rc = np.zeros(K)
            f1 = np.zeros(K)
            mcc = np.zeros(K)
            confMat = np.zeros(K, Nclasses, Nclasses)

            for i in range(K):
                _, acc[i], pr[i], rc[i], f1[i], mcc[i], confMat[i] = classification(XtrainList, Ytrain, XvalidList, Yvalid, XtestList, Ytest, Cs, 'mcc')


    else:
        _, acc, pr, rc, f1, mcc, confMat = classification(XtrainList, Ytrain, XvalidList, Yvalid, XtestList, Ytest, Cs, 'mcc')

    performance = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}


elif args.task == 'rank':
    # this should be the same as classification
    
    # a pickle file with the class label npy array with shape (N,)
    with open(args.YtrainFile, 'rb') as f:
        Ytrain = pickle.load(f)

    with open(args.XvalidFile, 'rb') as f:
        Yvalid = pickle.load(f)

    with open(args.XtestFile, 'rb') as f:
        Ytest = pickle.load(f)


    Nclasses = np.unique(Ytrain).shape[0]

    # MOFA, MoE, PoE, MVIB learn a common z --> X will be just an array N x F
    # CGAE has  a separate z for each modality --> so evaluate K times
    # baseline has the raw data and requires pca --> in both of these cases there will be a list of length K, with N x F matrices as elements

    if type(XtrainList) == list:
        if args.baseline:
            pcTrainList = []
            pcValidList = []
            pcTestList = []

            for i in range(XtrainList):
                pca = PCA(n_components=50, whiten=False, svd_solver='full')
                pca.fit(XtrainList[i])

                pcTrainList.append(pca.transform(XtrainList[i]))
                pcValidList.append(pca.transform(XvalidList[i]))
                pcTestList.append(pca.transform(XtestList[i]))

            Xtrain = np.hstack(pcTrainList)
            Xvalid = np.hstack(pcValidList)
            Xtest = np.hstack(pcTestList)

            assert Xtrain.shape[0] == Xtest.shape[0]
            assert Xtrain.shape[1] == 50 * len(XtrainList)

            _, acc, pr, rc, f1, mcc, confMat = ordinalRegression(XtrainList, Ytrain, XvalidList, Yvalid, XtestList, Ytest, Cs, 'mcc')

        else:
            K = len(XtrainList)
            assert len(XvalidList) == K
            assert len(XtestList) == K

            acc = np.zeros(K)
            pr = np.zeros(K)
            rc = np.zeros(K)
            f1 = np.zeros(K)
            mcc = np.zeros(K)
            confMat = np.zeros(K, Nclasses, Nclasses)

            for i in range(K):
                _, acc[i], pr[i], rc[i], f1[i], mcc[i], confMat[i] = ordinalRegression(XtrainList, Ytrain, XvalidList, Yvalid, XtestList, Ytest, Cs, 'mcc')


    else:
        _, acc, pr, rc, f1, mcc, confMat = ordinalRegression(XtrainList, Ytrain, XvalidList, Yvalid, XtestList, Ytest, Cs, 'mcc')

    performance = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}


else:
    assert args.task == 'impute'
    # other methods should generate the imputations by themselves
    # # TODO: not sure what will happen with MVIB here (ignore it?)
    assert args.baseline

    K = len(XtrainList)
    assert len(XvalidList) == K
    assert len(XtestList) == K

    # mse[i,j]: performance of using modality i to predict modality j
    mse = np.zeros((K, K), float)
    rsquared = np.eye(K)

    for i in range(K):
        for j in range(K):
            # loop over all pairs of data sources

            if i != j:
                _, (mse[i,j], rsquared[i,j]) = impute(XtrainList[i], XtrainList[j], XvalidList[i], XvalidList[j], XtestList[i], XtestList[j], alphas, 'mse')




    performance = {'mse': mse, 'rsquared': rsquared}




with open(args.resultsFile, 'wb') as f:
    pickle.dump(performance, f)
