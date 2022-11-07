import sys
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import torch
from src.nets import MLP
from src.util.early_stopping import EarlyStopping
import src.util.logger as logger
import os

class CustomDataset():
    def __init__(self, X, y):
        self.x = TensorDataset(torch.tensor(X))
        # load the 2nd data view
        self.y = TensorDataset(torch.LongTensor(y))
        self.length = y.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # get the index-th element of the 3 matrices
        return self.x.__getitem__(index), self.y.__getitem__(index)




if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Incorrect config.yaml file!")
            print(exc)


    logger.output_file = config['log']
    X = np.load(config['data_path'])

    trnInd = np.load(config['train_ind'])
    valInd = np.load(config['val_ind'])
    tstInd = np.load(config['test_ind'])

    y = np.load(config['labels'])

    yVal = y[valInd]
    yTest = y[tstInd]


    trainDataset = CustomDataset(X[trnInd], y[trnInd])
    validationDataset = CustomDataset(X[valInd], y[valInd])
    testDataset = CustomDataset(X[tstInd], y[tstInd])


    train_loader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=False)

    valid_loader = DataLoader(validationDataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, drop_last=False)

    test_loader = DataLoader(testDataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, drop_last=False)

    logger.info('defining model with %d classes' % np.unique(y).shape[0])
    model = MLP(X.shape[1], 64, np.unique(y).shape[0])
    model = model.double()

    device = torch.device('cuda:0')
    model = model.to(device)

    logger.success('Model ok')

    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], verbose=True)

    if not os.path.exists(config['checkpoint']):
        os.mkdir(config['checkpoint'])

    if not os.path.exists(config['log']):
        os.mkdir(config['log'])


    model.optimize(config['epochs'], config['lr'], train_loader, valid_loader, config['checkpoint'], config['log'], early_stopping)

    ypred_validation = np.zeros((valInd.shape[0],), int)
    ypred_test = np.zeros((tstInd.shape[0],), int)

    model.eval()
    with torch.no_grad():
        testLoss = 0.
        validationLoss = 0.

        loss_fun = torch.nn.CrossEntropyLoss()

        i = 0
        b = config['batch_size']
        for x, y in test_loader:
            y_pred = model.forward(x[0].double().to(device))

            ypred_test[i:i+b] = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
            testLoss += loss_fun(y_pred, y[0].to(device)).item() * x[0].shape[0]

            i += b

        testLoss /= len(train_loader.dataset)

        i = 0
        for x, y in valid_loader:
            y_pred = model.forward(x[0].double().to(device))

            ypred_validation[i:i+b] = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
            validationLoss += loss_fun(y_pred, y[0].to(device)).item()  * x[0].shape[0]

            i += b

        validationLoss /= len(valid_loader.dataset)

    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score


    logger.info('Validation performance:')
    logger.info('loss:\t%.3f' % validationLoss)
    logger.info('acc:\t%.3f' % accuracy_score(yVal, ypred_validation))
    logger.info('mcc:\t%.3f' % matthews_corrcoef(yVal, ypred_validation))
    logger.info('f1:\t%.3f' % f1_score(yVal, ypred_validation, average='macro'))

    logger.info('Test performance:')
    logger.info('loss:\t%.3f' % testLoss)
    logger.info('acc:\t%.3f' % accuracy_score(yTest, ypred_test))
    logger.info('mcc:\t%.3f' % matthews_corrcoef(yTest, ypred_test))
    logger.info('f1:\t%.3f' % f1_score(yTest, ypred_test, average='macro'))
