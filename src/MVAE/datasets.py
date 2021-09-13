import sys
import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from src.util import logger

TRAINING_DATA_SPLIT = 0.7
VALIDATION_DATA_SPLIT = 0.1
PREDICT_DATA_SPLIT = 0.2
VALID_PARTITIONS = {'train': 0, 'val': 1}


class TCGAData(object):
    """TCGA Landmarks dataset."""

    def __init__(self, args, save_dir=None, indices_path=None):
        """
        Datasets are assumed to be pre-processed and have the same ordering of samples

        Args:
            save_dir     (string) : Where the indices taken from the datasets should be saved
            indices_path (string) : If set, use predefined indices for data split
        """
        # Load in data depending on task
        if args['task'] == 1:
            # Task 1 : Imputation
            # variable y contains cancer type/cell type

            # Load in files for now
            self.ge_train_file = np.load(args['x_train_file'])
            self.ge_val_file = np.load(args['x_val_file'])
            self.ge_test_file = np.load(args['x_test_file'])
            self.me_train_file = np.load(args['y_train_file'])
            self.me_val_file = np.load(args['y_val_file'])
            self.me_test_file = np.load(args['y_test_file'])

            # GE = np.load(args['data_path1'])
            # ME = np.load(args['data_path2'])

            # assert GE.shape[0] == ME.shape[0], "Datasets do not have equal samples"

            # cancerType = np.load(args['cancer_type_index'])

            # split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

            # for trainValidInd, testInd in split1.split(GE, cancerType):
            #     # Get test split
            #     self.ge_test_file = np.float32(GE[testInd])
            #     self.me_test_file = np.float32(ME[testInd])
            #     cancerTypetest = cancerType[testInd]

            #     # Get training and validation splits
            #     GEtrainValid = GE[trainValidInd]
            #     MEtrainValid = ME[trainValidInd]
            #     cancerTypetrainValid = cancerType[trainValidInd]

            # split2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9)

            # for trainInd, validInd in split2.split(GEtrainValid, cancerTypetrainValid):
            #     # Train splits
            #     self.ge_train_file = np.float32(GEtrainValid[trainInd])
            #     self.me_train_file = np.float32(MEtrainValid[trainInd])
            #     cancerTypetrain = cancerTypetrainValid[trainInd]

            #     # Validation splits
            #     self.ge_val_file = np.float32(GEtrainValid[validInd])
            #     self.me_val_file = np.float32(MEtrainValid[validInd])
            #     cancerTypevalid = cancerTypetrainValid[validInd]

            # if save_dir is None:
            #     logger.error("Error, no save path is given so indices for data splits could not be saved. Exiting program")
            #     sys.exit()

            # # Save the indices taken for reproducibility
            # np.save("{}/training_indices.npy".format(save_dir), trainInd)
            # np.save("{}/validation_indices.npy".format(save_dir), validInd)
            # np.save("{}/test_indices.npy".format(save_dir), testInd)

        if args['task'] == 2:
            logger.success("Running Task 2: {} classification.".format(args['ctype']))
            # NOTE
            # For testing purposes, this code uses predefined splits, later this should be done everytime the model is run
            GEtrainctype = np.load(args['x_ctype_train_file'])
            GEtrainrest = np.load(args['x_train_file'])
            self.ge_train_file = np.float32(np.vstack((GEtrainctype, GEtrainrest)))

            GEvalidctype = np.load(args['x_ctype_valid_file'])
            GEvalidrest = np.load(args['x_valid_file'])
            self.ge_val_file = np.float32(np.vstack((GEvalidctype, GEvalidrest)))

            MEtrainctype = np.load(args['y_ctype_train_file'])
            MEtrainrest = np.load(args['y_train_file'])
            self.me_train_file = np.float32(np.vstack((MEtrainctype, MEtrainrest)))

            MEvalidctype = np.load(args['y_ctype_valid_file'])
            MEvalidrest = np.load(args['y_valid_file'])
            self.me_val_file = np.float32(np.vstack((MEvalidctype, MEvalidrest)))


    def get_data_partition(self, partition):
        if partition == "train":
            return TCGADataset(self.ge_train_file, self.me_train_file)
        elif partition == "val":
            return TCGADataset(self.ge_val_file, self.me_val_file)
        elif partition == "test":
            return TCGADataset(self.ge_test_file, self.me_test_file)
        else:  # Full data aka no split
            return TCGADataset(np.vstack((self.ge_train_file, self.ge_val_file, self.ge_test_file)),
                               np.vstack((self.me_train_file, self.me_val_file, self.me_test_file)))


class TCGADataset(Dataset):
    """
    Dataset Wrapper of TCGA Data (so data is loaded only once)
    """

    def __init__(self, ge_data, me_data):
        """
        Args:
            ge_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of RNA-seq
            me_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of Methylation
        """
        assert ge_data.shape[0] == me_data.shape[0], "Datasets do not have equal samples"

        self.ge_data = ge_data
        self.me_data = me_data

    def __len__(self):
        return self.ge_data.shape[0]

    def __getitem__(self, idx):
        return self.ge_data[idx], self.me_data[idx]
