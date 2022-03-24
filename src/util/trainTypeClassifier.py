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


y = np.load(config['labels'])

trainDataset = CustomDataset(X[trnInd], y[trnInd])
validationDataset = CustomDataset(X[valInd], y[valInd])


train_loader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=False)

valid_loader = DataLoader(validationDataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, drop_last=False)

logger.info('defining model')
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
