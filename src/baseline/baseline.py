import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from mord import OrdinalRidge
from src.baseline.evaluate import *
import src.util.logger as logger

def impute(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, criterion='mse'):

    # returns imputed values as well as evaluation of the imputation based on mse and rsquared

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = Ridge(alpha=a, fit_intercept=True, normalize=False, random_state=1)

        # train
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_imputation(y_valid, model.predict(x_valid), criterion)

    if criterion == 'mse':
        bestModel = models[np.argmin(validationPerformance)]
    else:
        bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test)

    return predictions, evaluate_imputation(y_test, predictions, 'mse'), evaluate_imputation(y_test, predictions, 'rsquared')


def ordinal_regression(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, criterion='mcc'):
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
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_classification(y_valid, model.predict(x_valid))[ind]

    bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test).astype(int)

    return predictions, evaluate_classification(y_test, predictions)


def classification(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, criterion='mcc'):
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
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_classification(y_valid, model.predict(x_valid))[ind]

    bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test).astype(int)

    return predictions, evaluate_classification(y_test, predictions)


def run_baseline(args: dict) -> None:
    """

    @param args: Dictionary containing input parameters
    @return:
    """
    save_dir = os.path.join(args['save_dir'], '{}'.format('baseline'))
    os.makedirs(save_dir)

    alphas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10., 20.])
    Cs = 1. / alphas

    print(args['task'])

    # load features (raw GE/ME/etc or VAE embeddings) from K data sources, N tumors, F features
    # a pickle file with a list of length K, each element containing a N x F npy array
    with open(args['x_train_file'], 'rb') as f:
        x_trainlist = np.float32(np.load(f, allow_pickle=True))

    with open(args['x_valid_file'], 'rb') as f:
        x_validlist = np.float32(np.load(f, allow_pickle=True))

    with open(args['x_test_file'], 'rb') as f:
        x_testlist = np.float32(np.load(f, allow_pickle=True))

    if args['task'] == 'classify':
        # a pickle file with the class label npy array with shape (N,)
        with open(args['y_train_file'], 'rb') as f:
            y_train = np.load(f, allow_pickle=True)

        with open(args['y_valid_file'], 'rb') as f:
            y_valid = np.load(f, allow_pickle=True)

        with open(args['y_test_file'], 'rb') as f:
            y_test = np.load(f, allow_pickle=True)

        Nclasses = np.unique(y_train).shape[0]

        # MOFA, MoE, PoE, MVIB learn a common z --> X will be just an array N x F
        # CGAE has  a separate z for each modality --> so evaluate K times
        # baseline has the raw data and requires pca --> in both of these cases there will be a list of length K, with N x F matrices as elements

        if type(x_trainlist) == list:
            if args['baseline']:
                pcTrainList = []
                pcValidList = []
                pcTestList = []

                for i in range(x_trainlist):
                    pca = PCA(n_components=50, whiten=False, svd_solver='full')
                    pca.fit(x_trainlist[i])

                    pcTrainList.append(pca.transform(x_trainlist[i]))
                    pcValidList.append(pca.transform(x_validlist[i]))
                    pcTestList.append(pca.transform(x_testlist[i]))

                x_train = np.hstack(pcTrainList)
                x_valid = np.hstack(pcValidList)
                x_test = np.hstack(pcTestList)

                assert x_train.shape[0] == x_test.shape[0]
                assert x_train.shape[1] == 50 * len(x_trainlist)

                _, acc, pr, rc, f1, mcc, confMat = classification(x_trainlist, y_train, x_validlist, y_valid, x_testlist,
                                                                  y_test, Cs, 'mcc')

            else:
                K = len(x_trainlist)
                assert len(x_validlist) == K
                assert len(x_testlist) == K

                acc = np.zeros(K)
                pr = np.zeros(K)
                rc = np.zeros(K)
                f1 = np.zeros(K)
                mcc = np.zeros(K)
                confMat = np.zeros(K, Nclasses, Nclasses)

                for i in range(K):
                    _, acc[i], pr[i], rc[i], f1[i], mcc[i], confMat[i] = classification(x_trainlist, y_train, x_validlist,
                                                                                        y_valid, x_testlist, y_test, Cs,
                                                                                        'mcc')


        else:
            _, acc, pr, rc, f1, mcc, confMat = classification(x_trainlist, y_train, x_validlist, y_valid, x_testlist, y_test,
                                                              Cs, 'mcc')

        performance = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}


    elif args['task'] == 'rank':
        # this should be the same as classification

        # a pickle file with the class label npy array with shape (N,)
        with open(args['y_train_file'], 'rb') as f:
            y_train = np.load(f, allow_pickle=True)

        with open(args['y_valid_file'], 'rb') as f:
            y_valid = np.load(f, allow_pickle=True)

        with open(args['y_test_file'], 'rb') as f:
            y_test = np.load(f, allow_pickle=True)

        Nclasses = np.unique(y_train).shape[0]

        # MOFA, MoE, PoE, MVIB learn a common z --> X will be just an array N x F
        # CGAE has  a separate z for each modality --> so evaluate K times
        # baseline has the raw data and requires pca --> in both of these cases there will be a list of length K, with N x F matrices as elements

        if type(x_trainlist) == list:
            if args['baseline']:
                pcTrainList = []
                pcValidList = []
                pcTestList = []

                for i in range(x_trainlist):
                    pca = PCA(n_components=50, whiten=False, svd_solver='full')
                    pca.fit(x_trainlist[i])

                    pcTrainList.append(pca.transform(x_trainlist[i]))
                    pcValidList.append(pca.transform(x_validlist[i]))
                    pcTestList.append(pca.transform(x_testlist[i]))

                x_train = np.hstack(pcTrainList)
                x_valid = np.hstack(pcValidList)
                x_test = np.hstack(pcTestList)

                assert x_train.shape[0] == x_test.shape[0]
                assert x_train.shape[1] == 50 * len(x_trainlist)

                _, acc, pr, rc, f1, mcc, confMat = ordinal_regression(x_trainlist, y_train, x_validlist, y_valid, x_testlist,
                                                                     y_test, Cs, 'mcc')

            else:
                K = len(x_trainlist)
                assert len(x_validlist) == K
                assert len(x_testlist) == K

                acc = np.zeros(K)
                pr = np.zeros(K)
                rc = np.zeros(K)
                f1 = np.zeros(K)
                mcc = np.zeros(K)
                confMat = np.zeros(K, Nclasses, Nclasses)

                for i in range(K):
                    _, acc[i], pr[i], rc[i], f1[i], mcc[i], confMat[i] = ordinal_regression(x_trainlist, y_train,
                                                                                           x_validlist, y_valid,
                                                                                           x_testlist, y_test, Cs, 'mcc')


        else:
            _, acc, pr, rc, f1, mcc, confMat = ordinal_regression(x_trainlist, y_train, x_validlist, y_valid, x_testlist,
                                                                 y_test, Cs, 'mcc')

        performance = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

    else:
        assert args['task'] == 'impute'
        logger.success("Running baseline for task {} with data from 1: {} and 2: {}".format(args['task'], args['data1'], args['data2']))
        # other methods should generate the imputations by themselves
        # # TODO: not sure what will happen with MVIB here (ignore it?)

        # a pickle file with the class label npy array with shape (N,)
        with open(args['y_train_file'], 'rb') as f:
            y_train = np.float32(np.load(f, allow_pickle=True))

        with open(args['y_valid_file'], 'rb') as f:
            y_valid = np.float32(np.load(f, allow_pickle=True))

        with open(args['y_test_file'], 'rb') as f:
            y_test = np.float32(np.load(f, allow_pickle=True))

        NR_MODALITIES = 2

        # mse[i,j]: performance of using modality i to predict modality j
        mse = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
        rsquared = np.eye(NR_MODALITIES)

        # From x to y
        _, mse[0, 1], rsquared[0, 1] = impute(x_trainlist, y_train, x_validlist, y_valid, x_testlist, y_test, alphas, 'mse')

        # From y to x
        _, mse[1, 0], rsquared[1, 0] = impute(y_train, x_trainlist, y_valid, x_validlist, y_test, x_testlist, alphas, 'mse')

        performance = {'mse': mse, 'rsquared': rsquared}

        logger.info("BASELINE RESULTS")
        logger.info("MSE: From {} to {} : {}".format(args['data1'], args['data2'], performance['mse'][0, 1]))
        logger.info("MSE: From {} to {} : {}".format(args['data2'], args['data1'], performance['mse'][1, 0]))
        logger.info("")
        logger.info("R^2 regression score function: From {} to {} : {}".format(args['data1'], args['data2'], performance['rsquared'][0, 1]))
        logger.info("R^2 regression score function: From {} to {} : {}".format(args['data2'], args['data1'], performance['rsquared'][1, 0]))

        print(performance)
        print(type(performance))

    with open(os.path.join(save_dir, args['name']), 'wb') as f:
        pickle.dump(performance, f)
