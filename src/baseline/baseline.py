import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.decomposition import PCA
from src.util.evaluate import *
import src.util.logger as logger
from src.util.umapplotter import UMAPPlotter
import matplotlib.pyplot as plt
import seaborn as sns

def impute(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, num_features, criterion='mse'):
    # returns imputed values as well as evaluation of the imputation based on mse and rsquared

    criteria = ['mse', 'spearman', 'r2']
    index = criteria.index(criterion)

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = Ridge(alpha=a, fit_intercept=True, random_state=1)
        # train
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_imputation(y_valid, model.predict(x_valid), num_features, criterion)[index]

    if criterion == 'mse':
        bestModel = models[np.argmin(validationPerformance)]
    else:
        bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test)

    mse, corr, rsq = evaluate_imputation(y_test, predictions, num_features, 'mse')


    return predictions, mse, corr, rsq



def classification(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, criterion='mcc', baseline=False):
    # returns the predicted class labels, as well as accuracy measures

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
        model = SGDClassifier(penalty='l2', loss='hinge', alpha=a, fit_intercept=True, random_state=1)

        # train
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_classification(y_valid, model.predict(x_valid))[ind]

    bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test).astype(int)
    acc, pr, rc, f1, mcc, confMat = evaluate_classification(y_test, predictions)
    return predictions, acc, pr, rc, f1, mcc, confMat


def run_baseline(args: dict) -> None:
    """

    @param args: Dictionary containing input parameters
    @return:
    """
    logger.info("Running Baseline - Task {}".format(args['task']))
    save_dir = os.path.join(args['save_dir'], '{}'.format('baseline'))
    os.makedirs(save_dir)

    alphas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10., 20.])


    # Load in data
    omic1 = np.load(args['data_path1']).astype(np.float32)
    omic2 = np.load(args['data_path2']).astype(np.float32)

    classLabels = np.load(args['labels'])
    labelNames = np.load(args['labelnames'])


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

    ytrain = classLabels[train_ind]
    yvalid = classLabels[val_ind]
    ytest = classLabels[test_ind]


    logger.info("Succesfully loaded in all data")
    if args['task'] > 0:
        logger.success("Running baseline for task {} with data from 1: {} and 2: {}".format(args['task'], args['data1'],
                                                                                            args['data2']))

        NR_MODALITIES = 2

        # mse[i,j]: performance of using modality i to predict modality j
        mse = np.zeros((NR_MODALITIES, NR_MODALITIES), float)
        rsquared = np.eye(NR_MODALITIES)
        spearman = np.zeros((NR_MODALITIES, NR_MODALITIES), float)  # ,2 since we report mean and median
        spearman_p = np.zeros((NR_MODALITIES, NR_MODALITIES), float)

        # From x to y
        omic2_from_omic1, mse[0, 1], spearman[0, 1], rsquared[0, 1] = impute(omic1_train_file, omic2_train_file, omic1_valid_file, omic2_valid_file, omic1_test_file, omic2_test_file, alphas, args['num_features2'], 'mse')
        # From y to x
        omic1_from_omic2, mse[1, 0], spearman[1, 0], rsquared[1, 0] = impute(omic2_train_file, omic1_train_file, omic2_valid_file, omic1_valid_file, omic2_test_file, omic1_test_file, alphas, args['num_features1'], 'mse')

        performance = {'mse': mse, 'rsquared': rsquared, 'spearman_corr': spearman, 'spearman_p': spearman_p}

        logger.info('Test performance, imputation error, modality 1')
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse[1, 0], spearman[1,0], rsquared[1,0]))

        logger.info('Test performance, imputation error, modality 2')
        logger.info('MSE: %.4f\tSpearman: %.4f\tR^2: %.4f' % (mse[0, 1], spearman[0,1], rsquared[0,1]))


    # also classify
    if args['task'] > 1:

        Nclasses = np.unique(ytrain).shape[0]


        pca1 = PCA(n_components=32, whiten=False, svd_solver='full')
        pca1.fit(omic1_train_file)

        X1train = pca1.transform(omic1_train_file)
        X1valid = pca1.transform(omic1_valid_file)
        X1test = pca1.transform(omic1_test_file)

        pca2 = PCA(n_components=32, whiten=False, svd_solver='full')
        pca2.fit(omic2_train_file)

        X2train = pca2.transform(omic2_train_file)
        X2valid = pca2.transform(omic2_valid_file)
        X2test = pca2.transform(omic2_test_file)


        Xtrain = np.hstack((X1train, X2train))
        Xvalid = np.hstack((X1valid, X2valid))
        Xtest = np.hstack((X1test, X2test))

        assert Xtrain.shape[1] == Xtest.shape[1]
        # assert x_train.shape[1] == 50 * len(omic1_train_file)

        _, acc, pr, rc, f1, mcc, confMat = classification(X1train, ytrain, X1valid, yvalid, X1test, ytest, alphas, args['clf_criterion'])

        logger.info('Test performance, classification task, modality 1')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
        performanceClf1 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

        _, acc, pr, rc, f1, mcc, confMat = classification(X2train, ytrain, X2valid, yvalid, X2test, ytest, alphas, args['clf_criterion'])

        logger.info('Test performance, classification task, modality 2')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))

        performanceClf2 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}
        _, acc, pr, rc, f1, mcc, confMat = classification(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, alphas, args['clf_criterion'])

        performanceClf12 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat}

        logger.info('Test performance, classification task, both modalities')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))

        performance['omic1'] = performanceClf1
        performance['omic2'] = performanceClf2
        performance['omic1+2'] = performanceClf12

    with open(os.path.join(save_dir, args['name'] + 'results_pickle'), 'wb') as f:
        pickle.dump(performance, f)
