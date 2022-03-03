from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_recall_fscore_support, \
    matthews_corrcoef, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import stats
import numpy as np
import pandas as pd
import math


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

    # if criterion == 'mse':
    #     # For MSE feature-or-sample-wise does not matter
    #     return mean_squared_error(y_true, y_pred, multioutput=aggregate)
    #
    # elif criterion == 'spearman_corr':
    #     # Spearman Correlation gives NaN values, or will simply require >20GB memory on large datasets.
    #     # Therefore, do manual column wise (feature) correlation.
    #     correlations = np.zeros(num_features)
    #
    #     # Above it has been assured that the inputs are of shape (n_samples, n_observations)
    #     for i in range(y_true.shape[1]):
    #         corr = stats.spearmanr(y_true[:, i], y_pred[:, i])[0]  # [0] is correlation
    #         if math.isnan(corr):
    #             corr = 0
    #         correlations[i] = corr
    #
    #     # To see the influence of NaN on the correlation, report mean and median
    #     return [np.mean(correlations), np.median(correlations)]
    #
    # elif criterion == 'spearman_p':
    #     # Spearman Correlation gives NaN values, or will simply require >20GB memory on large datasets.
    #     # Therefore, do manual column wise (feature) correlation.
    #     p_values = np.zeros(num_features)
    #
    #     # Above it has been assured that the inputs are of shape (n_samples, n_observations)
    #     for i in range(y_true.shape[1]):
    #         corr = stats.spearmanr(y_true[:, i], y_pred[:, i])[1]  # [1] is p_value
    #         if math.isnan(corr):
    #             corr = 0
    #         p_values[i] = corr
    #
    #     return np.mean(p_values)
    #
    # else:
    #     # Do feature-wise Pearson correlation
    #     pearson_corrs = np.zeros(num_features)
    #
    #     # Above it has been assured that the inputs are of shape (n_samples, n_observations)
    #     for i in range(y_true.shape[1]):
    #         corr = r2_score(y_true[:, i], y_pred[:, i], multioutput=aggregate)
    #         pearson_corrs[i] = corr
    #
    #     return np.mean(pearson_corrs)


def evaluate_classification(y_true, y_pred):
    # returns accuracy, precision, recall, f1, mcc, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    confMat = confusion_matrix(y_true, y_pred)

    return [acc, pr, rc, f1, mcc, confMat]

def evaluate_generation(X1, X2, y, reconstructed1, reconstructed2):
    clf1 = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
    clf1.fit(X1, y)

    clf2 = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
    clf2.fit(X2, y)

    p1 = clf1.predict(reconstructed1)
    p2 = clf2.predict(reconstructed2)

    return accuracy_score(p1, p2)





def save_factorizations_to_csv(z, sample_names, save_dir, name):
    df = pd.DataFrame(z)

    if df.shape[0] != len(sample_names):
        df = df.transpose()

    df.index = list(sample_names)
    df.to_csv(path_or_buf="{}/{}.csv".format(save_dir, name), index=True)
