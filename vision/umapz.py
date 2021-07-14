import torch

import umap
import umap.plot
from train import load_checkpoint
import datasets
import numpy as np
import pandas as pd

if __name__ == "__main__":
    print("-----   Loading Trained Model   -----")
    model_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE cancer3 14-06-2021 09:09:39/best_model.pth.tar"
    model = load_checkpoint(model_path, use_cuda=False)
    model.eval()

    cancer3types = False
    if cancer3types:
        print("Umap of 3 cancer type model")
        train_dataset = [
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_training.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_training.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_training.csv",
                            index_col=0).to_numpy())
        ]

        val_dataset = [
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_validation.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_validation.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_validation.csv",
                            index_col=0).to_numpy())
        ]

        predict_dataset = [
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_predict.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_predict.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_predict.csv",
                            index_col=0).to_numpy())
        ]

        rna_data = np.vstack((train_dataset[0], val_dataset[0], predict_dataset[0]))
        gcn_data = np.vstack((train_dataset[1], val_dataset[1], predict_dataset[1]))
        dna_data = np.vstack((train_dataset[2], val_dataset[2], predict_dataset[2]))

        labels_train = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_training_3types.npy")
        labels_val = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_validation_3types.npy")
        labels_predict = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_predict_3types.npy")
        cancer_types_samples = np.concatenate((labels_train, labels_val, labels_predict))

        full_dataset = datasets.TCGADataset(rna_data, gcn_data, dna_data)

        # 1 batch for all data
        data_loader = torch.utils.data.DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)

    else:
        indices_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE shuffle 14-06-2021 08:24:33"
        tcga_data = datasets.TCGAData(indices_path=indices_path)
        cancer_types_samples = np.load("/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_cancertypes_samples.npy")

        training_ids = np.load("{}/training_indices.npy".format(indices_path))
        predict_ids = np.load("{}/predict_indices.npy".format(indices_path))

        full_dataset = tcga_data.get_data_partition("all")

        # 1 batch for all data
        data_loader = torch.utils.data.DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)

    # Get Latent Space
    print("Entering Data into model, fetching Z and fitting UMAP.")
    z = 0
    for batch_idx, (rna, gcn, dna) in enumerate(data_loader):
        mu, logvar = model.get_params(rna=rna, gcn=gcn, dna=dna)
        z = model.reparameterize(mu, logvar)

    # Transform z to numpy
    z = z.detach().numpy()

    mapper = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean'
    ).fit(z)

    print("Umap has been fit. Now plot")
    
    p = umap.plot.points(mapper, labels=cancer_types_samples)
    umap.plot.plt.title(
        "Product of Experts Latent Space UMAP : Colored per cancer type: \nEpochs : 100, Batch size : 256, Latent space : 128, LR: 0.0001")
    # umap.plot.plt.legend()
    umap.plot.plt.show()
