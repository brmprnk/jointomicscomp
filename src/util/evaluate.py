from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_recall_fscore_support, \
    matthews_corrcoef, confusion_matrix
from scipy.stats import stats
import numpy as np
import pandas as pd


def evaluate_imputation(y_true, y_pred, criterion, aggregate='uniform_average'):
    # returns mse and r squared, by default averages over all features
    if criterion == 'mse':
        return mean_squared_error(y_true, y_pred, multioutput=aggregate)
    elif criterion == 'spearman_corr':
        # correlation, p-value
        spear = stats.spearmanr(y_true, y_pred)
        correlation = spear[0]
        if isinstance(correlation, np.float64):
            return correlation
        else:
            return np.mean(correlation)
    elif criterion == 'spearman_p':
        # correlation, p-value
        spear = stats.spearmanr(y_true, y_pred)
        p_value = spear[1]
        if isinstance(p_value, np.float64):
            return p_value
        else:
            return np.mean(p_value)
    else:
        return r2_score(y_true, y_pred, multioutput=aggregate)


def evaluate_classification(y_true, y_pred):
    # returns accuracy, precision, recall, f1, mcc, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    confMat = confusion_matrix(y_true, y_pred)

    return [acc, pr, rc, f1, mcc, confMat]


def save_factorizations_to_csv(z, sample_names, save_dir, name):
    df = pd.DataFrame(z)

    if df.shape[0] != len(sample_names):
        df = df.transpose()

    df.index = list(sample_names)
    df.to_csv(path_or_buf="{}/{}.csv".format(save_dir, name), index=True)

