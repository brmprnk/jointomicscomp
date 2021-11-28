from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix


def evaluate_imputation(y_true, y_pred, criterion, aggregate='uniform_average'):
    # returns mse and r squared, by default averages over all features
    if criterion == 'mse':
        return mean_squared_error(y_true, y_pred, multioutput=aggregate)
    else:
        return r2_score(y_true, y_pred, multioutput=aggregate)


def evaluate_classification(y_true, y_pred):
    # returns accuracy, precision, recall, f1, mcc, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    confMat = confusion_matrix(y_true, y_pred)

    return [acc, pr, rc, f1, mcc, confMat]
