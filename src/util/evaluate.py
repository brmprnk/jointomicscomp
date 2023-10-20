from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_recall_fscore_support, \
    matthews_corrcoef, confusion_matrix
from sklearn.utils import resample
from sklearn.utils.random import sample_without_replacement
from scipy.stats import stats
import numpy as np
import pandas as pd
import math
from src.nets import MLP
import torch

def evaluate_imputation(y_true, y_pred, num_features, criterion='mse', aggregate='uniform_average'):
    """
    returns mse r squared or spearman, by default averages over all features

    @param y_true:        Input data
    @param y_pred:        Predicted data
    @param num_features   Required to ensure the input shapes have the features on the columns!
                          Easier than num_samples, since num_features is a required field in the config files.
    @param criterion:     Measure between input and predicted data, mse, pearson or spearman_corr / spearman_p\
    @param aggregate:     For sklearn default value
    :
    @return:
    """
    # Ensure features are on the columns (do feature-wise computations)
    # y_pred and y_true have to be of shape (n_samples, n_observations)
    if y_true.shape[1] != num_features:
        y_true = y_true.T
    if y_pred.shape[1] != num_features:
        y_pred = y_pred.T

    assert y_true.shape[1] == num_features, "Features are not on the rows for y_true in {} evaluation.".format(criterion)
    assert y_pred.shape[1] == num_features, "Features are not on the rows for y_pred in {} evaluation.".format(criterion)

    mse = mean_squared_error(y_true, y_pred, multioutput=aggregate)
    correlations = np.zeros(num_features)

    # Above it has been assured that the inputs are of shape (n_samples, n_observations)
    for i in range(y_true.shape[1]):
        corr = stats.spearmanr(y_true[:, i], y_pred[:, i])[0]  # [0] is correlation
        if math.isnan(corr):
            corr = 0
        correlations[i] = corr

    rsquared = r2_score(y_true, y_pred, multioutput=aggregate)

    return mse, np.median(correlations), rsquared


def evaluate_classification(y_true, y_pred, Nbootstraps=100, bootstrapSeed=1):
    # returns accuracy, precision, recall, f1, mcc, confusion_matrix and bootstraps

    seeds = sample_without_replacement(n_population=2**32, n_samples=Nbootstraps, random_state=bootstrapSeed, method='auto')

    [acc, pr, rc, f1, mcc, confMat] = calculate_classification_metrics(y_true, y_pred)

    performances = np.zeros((Nbootstraps, 5))

    for i, seed in enumerate(seeds):
        randomTrue, randomPred = resample(y_true, y_pred, replace=True, random_state=seed)
        [accR, prR, rcR, f1R, mccR] = calculate_classification_metrics(randomTrue, randomPred)[:5]

        performances[i] = [accR, np.mean(prR), np.mean(rcR), np.mean(f1R), mccR]

    # acc = accuracy_score(y_true, y_pred)
    # pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    # mcc = matthews_corrcoef(y_true, y_pred)
    # confMat = confusion_matrix(y_true, y_pred)
    CIs = np.percentile(performances, [2.5, 97.5], axis=0)

    return [acc, pr, rc, f1, mcc, confMat, CIs]



def calculate_classification_metrics(y_true, y_pred):
    # returns accuracy, precision, recall, f1, mcc, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    confMat = confusion_matrix(y_true, y_pred)

    return [acc, pr, rc, f1, mcc, confMat]



def evaluate_generation(reconstructed1, reconstructed2, datatype1, datatype2):
    if datatype2 == 'ATAC':
        assert datatype1 == 'RNA'
        datatype1 = 'RNA2'
        nclass = 7

    elif datatype1 == 'RNA' or datatype1 == 'ADT':
        nclass = 30
    else:
        nclass = 33

    clf1 = MLP(reconstructed1.shape[1], 64, nclass).double()
    clf2 = MLP(reconstructed2.shape[1], 64, nclass).double()

    checkpoint = torch.load('type-classifier/' + datatype1 + '/checkpoint/model_best.pth.tar')
    clf1.load_state_dict(checkpoint['state_dict'])

    checkpoint = torch.load('type-classifier/' + datatype2 + '/checkpoint/model_best.pth.tar')
    clf2.load_state_dict(checkpoint['state_dict'])


    y1 = clf1.predict(reconstructed1.double())
    y2 = clf2.predict(reconstructed2.double())

    p1 = torch.argmax(y1, axis=1)
    p2 = torch.argmax(y2, axis=1)

    return accuracy_score(p1, p2)



def save_factorizations_to_csv(z, sample_names, save_dir, name):
    df = pd.DataFrame(z)

    if df.shape[0] != len(sample_names):
        df = df.transpose()

    df.index = list(sample_names)
    df.to_csv(path_or_buf="{}/{}.csv".format(save_dir, name), index=True)
