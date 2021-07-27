from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix



def evaluate_imputation(Ytrue, Ypred, aggregate='uniform_average'):
    # returns mse and r squared, by default averages over all features

    return mean_squared_error(Ytrue, Ypred, multioutput=aggregate), r2_score(Ytrue, Ypred, multioutput=aggregate)


def evaluate_classification(Ytrue, Ypred):
    #retruns accuracy, precision, recall, f1, mcc, confusion_matrix

    acc = accuracy_score(Ytrue, Ypred)
    pr, rc, f1, _ = precision_recall_fscore_support(Ytrue, Ypred)
    mcc = matthews_corrcoef(Ytrue, Ypred)
    confMat = confusion_matrix(Ytrue, Ypred)

    return [acc, pr, rc, f1, mcc, confMat]
