from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
from torch.utils.data.dataset import Dataset

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
        print(args)
        # GE
        self.GE = np.load(args['data_path1'])
        print("-----   GE file read   -----")

        # ME
        self.ME = np.load(args['data_path2'])
        print("-----   ME file read   -----")

        # Split Datasets into a 70 training / 10 validation / 20 prediction split
        assert self.GE.shape[0] == self.ME.shape[0], "Datasets do not have equal samples"

        # Split the dataset
        if indices_path is None:
            nr_of_samples = self.GE.shape[0]
            nr_of_training_samples = int(TRAINING_DATA_SPLIT * nr_of_samples)
            nr_of_validation_samples = int(VALIDATION_DATA_SPLIT * nr_of_samples)

            # Random ordering of all sample id's
            random_sample_indices = np.random.choice(a=nr_of_samples, size=nr_of_samples, replace=False)

            # Split into three sets of sizes
            # [:nr_of_training_samples], [nr_of_training_samples:nr_of_validation_samples], [:nr_of_predict_samples]
            sets = np.split(random_sample_indices,
                            [nr_of_training_samples, (nr_of_training_samples + nr_of_validation_samples)])

            training_ids = sets[0]
            validation_ids = sets[1]
            predict_ids = sets[2]

            if save_dir is None:
                print("Error, no save path is given so indices for data splits could not be saved. Exiting program")
                sys.exit()

            # Save the indices taken for reproducibility
            np.save("{}/training_indices.npy".format(save_dir), training_ids)
            np.save("{}/validation_indices.npy".format(save_dir), validation_ids)
            np.save("{}/predict_indices.npy".format(save_dir), predict_ids)

        else:  # Use predefined indices
            print("Using Predefined split")
            training_ids = np.load("{}/training_indices.npy".format(indices_path))
            validation_ids = np.load("{}/validation_indices.npy".format(indices_path))
            predict_ids = np.load("{}/predict_indices.npy".format(indices_path))

        # Create data arrays
        self.ge_train_file = np.float32(self.GE[training_ids])
        self.ge_val_file = np.float32(self.GE[validation_ids])
        self.ge_predict_file = np.float32(self.GE[predict_ids])

        self.me_train_file = np.float32(self.ME[training_ids])
        self.me_val_file = np.float32(self.ME[validation_ids])
        self.me_predict_file = np.float32(self.ME[predict_ids])

    def get_data_partition(self, partition):
        if partition == "train":
            return TCGADataset(self.ge_train_file, self.me_train_file)
        elif partition == "val":
            return TCGADataset(self.ge_val_file, self.me_val_file)
        elif partition == "predict":
            return TCGADataset(self.ge_predict_file, self.me_predict_file)
        else:  # Full data aka no split
            return TCGADataset(self.GE, self.ME)


class TCGADataset(Dataset):
    """
    Dataset Wrapper of TCGA Data (so data is loaded only once)
    """

    def __init__(self, ge_data, me_data):
        """
        Args:
            rna_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of RNA-seq
            gcn_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of Gene Copy Number
        """
        assert ge_data.shape[0] == me_data.shape[0], "Datasets do not have equal samples"

        self.ge_data = ge_data
        self.me_data = me_data

    def __len__(self):
        return self.ge_data.shape[0]

    def __getitem__(self, idx):
        return self.ge_data[idx], self.me_data[idx]
