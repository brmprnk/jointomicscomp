import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from mord import OrdinalRidge
from src.util.evaluate import *
import src.util.logger as logger
from src.util.umapplotter import UMAPPlotter


def impute(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, num_features, criterion='mse'):
    # returns imputed values as well as evaluation of the imputation based on mse and rsquared

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = Ridge(alpha=a, fit_intercept=True, random_state=1)

        # train
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_imputation(y_valid, model.predict(x_valid), num_features, criterion)

    if criterion == 'mse':
        bestModel = models[np.argmin(validationPerformance)]
    else:
        bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test)

    return predictions, \
           evaluate_imputation(y_test, predictions, num_features, 'mse'), \
           evaluate_imputation(y_test, predictions, num_features, 'rsquared'), \
           evaluate_imputation(y_test, predictions, num_features, 'spearman_corr'), \
           evaluate_imputation(y_test, predictions, num_features, 'spearman_p')


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
        model = OrdinalRidge(alpha=a, fit_intercept=True, random_state=1)

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
    logger.info("Running Baseline - Task {}".format(args['task']))
    save_dir = os.path.join(args['save_dir'], '{}'.format('baseline'))
    os.makedirs(save_dir)

    alphas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10., 20.])
    Cs = 1. / alphas

    # Load in data
    omic1 = np.load(args['data_path1']).astype(np.float32)
    omic2 = np.load(args['data_path2']).astype(np.float32)

    # Use predefined split
    train_ind = np.load(args['train_ind'])
    val_ind = np.load(args['val_ind'])
    test_ind = np.load(args['test_ind'])

    # load features (raw GE/ME/etc or VAE embeddings) from K data sources, N tumors, F features
    # a pickle file with a list of length K, each element containing a N x F npy array
    omic1_train_file = omic1[train_ind]
    omic1_valid_file = omic1[val_ind]
    omic1_test_file = omic1[test_ind]

    omic2_train_file = omic2[train_ind]
    omic2_valid_file = omic2[val_ind]
    omic2_test_file = omic2[test_ind]

    logger.info("Succesfully loaded in all data")

    if args['task'] == 'classify':

        Nclasses = np.unique(omic2_train_file).shape[0]

        # MOFA, MoE, PoE, MVIB learn a common z --> X will be just an array N x F
        # CGAE has  a separate z for each modality --> so evaluate K times
        # baseline has the raw data and requires pca --> in both of these cases there will be a list of length K, with N x F matrices as elements

        if type(omic1_train_file) == list:
            if args['baseline']:
                pcTrainList = []
                pcValidList = []
                pcTestList = []

                for i in range(omic1_train_file):
                    pca = PCA(n_components=50, whiten=False, svd_solver='full')
                    pca.fit(omic1_train_file[i])

                    pcTrainList.append(pca.transform(omic1_train_file[i]))
                    pcValidList.append(pca.transform(omic1_valid_file[i]))
                    pcTestList.append(pca.transform(omic1_test_file[i]))

                x_train = np.hstack(pcTrainList)
                x_valid = np.hstack(pcValidList)
                x_test = np.hstack(pcTestList)

                assert x_train.shape[0] == x_test.shape[0]
                assert x_train.shape[1] == 50 * len(omic1_train_file)

                _, acc, pr, rc, f1, mcc, confMat = classification(omic1_train_file, omic2_train_file, omic1_valid_file,
                                                                  omic2_valid_file, omic1_test_file,
                                                                  omic2_test_file, Cs, 'mcc')

            else:
                K = len(omic1_train_file)
                assert len(omic1_valid_file) == K
                assert len(omic1_test_file) == K

                acc = np.zeros(K)
                pr = np.zeros(K)
                rc = np.zeros(K)
                f1 = np.zeros(K)
                mcc = np.zeros(K)
                confMat = np.zeros(K, Nclasses, Nclasses)

                for i in range(K):
                    _, acc[i], pr[i], rc[i], f1[i], mcc[i], confMat[i] = classification(omic1_train_file,
                                                                                        omic2_train_file,
                                                                                        omic1_valid_file,
                                                                                        omic2_valid_file,
                                                                                        omic1_test_file,
                                                                                        omic2_test_file, Cs,
                                                                                        'mcc')


        else:
            _, acc, pr, rc, f1, mcc, confMat = classification(omic1_train_file, omic2_train_file, omic1_valid_file,
                                                              omic2_valid_file, omic1_test_file,
                                                              omic2_test_file,
                                                              Cs, 'mcc')

        performance = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}


    elif args['task'] == 'rank':
        # this should be the same as classification

        Nclasses = np.unique(omic2_train_file).shape[0]

        # MOFA, MoE, PoE, MVIB learn a common z --> X will be just an array N x F
        # CGAE has  a separate z for each modality --> so evaluate K times
        # baseline has the raw data and requires pca --> in both of these cases there will be a list of length K, with N x F matrices as elements

        if type(omic1_train_file) == list:
            if args['baseline']:
                pcTrainList = []
                pcValidList = []
                pcTestList = []

                for i in range(omic1_train_file):
                    pca = PCA(n_components=50, whiten=False, svd_solver='full')
                    pca.fit(omic1_train_file[i])

                    pcTrainList.append(pca.transform(omic1_train_file[i]))
                    pcValidList.append(pca.transform(omic1_valid_file[i]))
                    pcTestList.append(pca.transform(omic1_test_file[i]))

                x_train = np.hstack(pcTrainList)
                x_valid = np.hstack(pcValidList)
                x_test = np.hstack(pcTestList)

                assert x_train.shape[0] == x_test.shape[0]
                assert x_train.shape[1] == 50 * len(omic1_train_file)

                _, acc, pr, rc, f1, mcc, confMat = ordinal_regression(omic1_train_file, omic2_train_file,
                                                                      omic1_valid_file, omic2_valid_file,
                                                                      omic1_test_file,
                                                                      omic2_test_file, Cs, 'mcc')

            else:
                K = len(omic1_train_file)
                assert len(omic1_valid_file) == K
                assert len(omic1_test_file) == K

                acc = np.zeros(K)
                pr = np.zeros(K)
                rc = np.zeros(K)
                f1 = np.zeros(K)
                mcc = np.zeros(K)
                confMat = np.zeros(K, Nclasses, Nclasses)

                for i in range(K):
                    _, acc[i], pr[i], rc[i], f1[i], mcc[i], confMat[i] = ordinal_regression(omic1_train_file,
                                                                                            omic2_train_file,
                                                                                            omic1_valid_file,
                                                                                            omic2_valid_file,
                                                                                            omic1_test_file,
                                                                                            omic2_test_file, Cs, 'mcc')


        else:
            _, acc, pr, rc, f1, mcc, confMat = ordinal_regression(omic1_train_file, omic2_train_file, omic1_valid_file,
                                                                  omic2_valid_file, omic1_test_file,
                                                                  omic2_test_file, Cs, 'mcc')

        performance = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

    else:
        assert args['task'] == 'impute'
        logger.success("Running baseline for task {} with data from 1: {} and 2: {}".format(args['task'], args['data1'],
                                                                                            args['data2']))
        # other methods should generate the imputations by themselves
        # # TODO: not sure what will happen with MVIB here (ignore it?)

        NR_MODALITIES = 2

        # mse[i,j]: performance of using modality i to predict modality j
        mse = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
        rsquared = np.eye(NR_MODALITIES)
        spearman = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
        spearman_p = np.zeros((NR_MODALITIES, NR_MODALITIES), float)

        # From x to y
        omic2_from_omic1, mse[0, 1], rsquared[0, 1], spearman[0, 1], spearman_p[0, 1] =\
            impute(omic1_train_file, omic2_train_file, omic1_valid_file,
                   omic2_valid_file, omic1_test_file, omic2_test_file, alphas, args['num_features2'], 'mse')

        # From y to x
        omic1_from_omic2, mse[1, 0], rsquared[1, 0], spearman[1, 0], spearman_p[1, 0] = \
            impute(omic2_train_file, omic1_train_file, omic2_valid_file,
                   omic1_valid_file, omic2_test_file, omic1_test_file, alphas, args['num_features1'], 'mse')

        performance = {'mse': mse, 'rsquared': rsquared, 'spearman_corr': spearman, 'spearman_p': spearman_p}

        logger.info("BASELINE RESULTS")
        logger.info("MSE: From {} to {} : {}".format(args['data1'], args['data2'], performance['mse'][0, 1]))
        logger.info("MSE: From {} to {} : {}".format(args['data2'], args['data1'], performance['mse'][1, 0]))
        logger.info("")
        logger.info("R^2 regression score function: From {} to {} : {}".format(args['data1'], args['data2'],
                                                                               performance['rsquared'][0, 1]))
        logger.info("R^2 regression score function: From {} to {} : {}".format(args['data2'], args['data1'],
                                                                               performance['rsquared'][1, 0]))
        logger.info("")
        logger.info("SPEARMAN CORR: From {} to {} : {}".format(args['data1'], args['data2'], performance['spearman_corr'][0, 1]))
        logger.info("SPEARMAN CORR: From {} to {} : {}".format(args['data2'], args['data1'], performance['spearman_corr'][1, 0]))
        logger.info("SPEARMAN P-value: From {} to {} : {}".format(args['data1'], args['data2'], performance['spearman_p'][0, 1]))
        logger.info("SPEARMAN P-value: From {} to {} : {}".format(args['data2'], args['data1'], performance['spearman_p'][1, 0]))

        print(performance)

    with open(os.path.join(save_dir, args['name'] + 'results_pickle'), 'wb') as f:
        pickle.dump(performance, f)
