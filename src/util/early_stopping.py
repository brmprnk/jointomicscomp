import numpy as np
import src.util.logger as logger
import torch


# from github Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0.5, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum % change in the monitored quantity to qualify as an improvement.
                            Default: 1
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta / 100  # as a percentage
        self.path = path

    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score

        if self.best_score > 0:
            deltaSign = 1.
        else:
            deltaSign = -1.

        if isinstance(score, torch.Tensor):
            if torch.gt(score, (self.best_score - deltaSign * (self.best_score * self.delta))):
                self.counter += 1
                logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > (self.best_score - deltaSign * (self.best_score * self.delta)):
                self.counter += 1
                logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
