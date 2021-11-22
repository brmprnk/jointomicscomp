import numpy as np
from torch.utils.data.dataset import Dataset
from src.util import logger


class TCGAData(object):
    """TCGA Landmarks dataset."""

    def __init__(self, args, save_dir=None, indices_path=None):
        """
        Datasets are assumed to be pre-processed and have the same ordering of samples

        Args:
            save_dir     (string) : Where the indices taken from the datasets should be saved
            indices_path (string) : If set, use predefined indices for data split
        """
        # Load in data, depending on task
        # Task 1 : Imputation
        if args['task'] == 1:
            logger.info("Running Task {} on omic {} and omic {}".format(args['task'], args['data1'], args['data2']))

            # Load in data
            omic1 = np.load(args['data_path1']).astype(np.float64)
            omic2 = np.load(args['data_path2']).astype(np.float64)
            self.labels = np.load(args['labels'])
            self.labelnames = np.load(args['labelnames'])

            # Use predefined split
            self.train_ind = np.load(args['train_ind'])
            self.val_ind = np.load(args['val_ind'])
            self.test_ind = np.load(args['test_ind'])

            self.omic1_train_file = omic1[self.train_ind]
            self.omic1_val_file = omic1[self.val_ind]
            self.omic1_test_file = omic1[self.test_ind]
            self.omic2_train_file = omic2[self.train_ind]
            self.omic2_val_file = omic2[self.val_ind]
            self.omic2_test_file = omic2[self.test_ind]

        if args['task'] == 2:
            logger.success("Running Task 2: {} classification.".format(args['ctype']))
            # NOTE
            # For testing purposes, this code uses predefined splits
            GEtrainctype = np.load(args['x_ctype_train_file'])
            GEtrainrest = np.load(args['x_train_file'])
            self.omic1_train_file = np.float32(np.vstack((GEtrainctype, GEtrainrest)))

            GEvalidctype = np.load(args['x_ctype_valid_file'])
            GEvalidrest = np.load(args['x_valid_file'])
            self.omic1_val_file = np.float32(np.vstack((GEvalidctype, GEvalidrest)))

            MEtrainctype = np.load(args['y_ctype_train_file'])
            MEtrainrest = np.load(args['y_train_file'])
            self.omic2_train_file = np.float32(np.vstack((MEtrainctype, MEtrainrest)))

            MEvalidctype = np.load(args['y_ctype_valid_file'])
            MEvalidrest = np.load(args['y_valid_file'])
            self.omic2_val_file = np.float32(np.vstack((MEvalidctype, MEvalidrest)))

    def get_data_partition(self, partition):
        if partition == "train":
            return TCGADataset(self.omic1_train_file, self.omic2_train_file)
        elif partition == "val":
            return TCGADataset(self.omic1_val_file, self.omic2_val_file)
        elif partition == "test":
            return TCGADataset(self.omic1_test_file, self.omic2_test_file)
        else:  # Full data aka no split
            return TCGADataset(np.vstack((self.omic1_train_file, self.omic1_val_file, self.omic1_test_file)),
                               np.vstack((self.omic2_train_file, self.omic2_val_file, self.omic2_test_file)))

    def get_labels_partition(self, partition):
        if partition == "train":
            return self.labels, self.labelnames, self.train_ind
        elif partition == "val":
            return self.labels, self.labelnames, self.val_ind
        elif partition == "test":
            return self.labels, self.labelnames, self.test_ind
        else:  # Full data aka no split
            return self.labels, self.labelnames


class TCGADataset(Dataset):
    """
    Dataset Wrapper of TCGA Data (so data is loaded only once)
    """

    def __init__(self, omic1_data, omic2_data):
        """
        Args:
            omic1_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of RNA-seq
            omic2_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of Methylation
        """
        assert omic1_data.shape[0] == omic2_data.shape[0], "Datasets do not have equal samples"

        self.omic1_data = omic1_data
        self.omic2_data = omic2_data

    def __len__(self):
        return self.omic1_data.shape[0]

    def __getitem__(self, idx):
        return self.omic1_data[idx], self.omic2_data[idx]
