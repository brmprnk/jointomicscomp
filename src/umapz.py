import os
import torch
import umap.plot
from train import load_checkpoint
import datasets
import numpy as np
import pandas as pd


class UMAP:

    def __init__(self, cancer3):
        self.cancer3types = cancer3

    def plot_umap(self):
        print("-----   Loading Trained Model   -----")
        results_dir = "/Users/bram/bachelor-thesis/results/followup_alldata 15-07-2021 14:47:17"
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

        model, checkpoint = load_checkpoint(os.path.join(results_dir, 'best_model.pth.tar'), use_cuda=False)
        model.eval()

        if self.cancer3types:
            print("Umap of 3 cancer type model")
            train_dataset = [
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'RNASeq_3types_training.csv'),
                                index_col=0).to_numpy()),
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'GCN_3types_training.csv'),
                                index_col=0).to_numpy()),
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'DNAMe_3types_training.csv'),
                                index_col=0).to_numpy())
            ]

            val_dataset = [
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'RNASeq_3types_validation.csv'),
                                index_col=0).to_numpy()),
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'GCN_3types_validation.csv'),
                                index_col=0).to_numpy()),
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'DNAMe_3types_validation.csv'),
                                index_col=0).to_numpy())
            ]

            predict_dataset = [
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'RNASeq_3types_predict.csv'),
                                index_col=0).to_numpy()),
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'GCN_3types_predict.csv'),
                                index_col=0).to_numpy()),
                np.float32(
                    pd.read_csv(os.path.join(data_dir, '3types', 'DNAMe_3types_predict.csv'),
                                index_col=0).to_numpy())
            ]

            rna_data = np.vstack((train_dataset[0], val_dataset[0], predict_dataset[0]))
            gcn_data = np.vstack((train_dataset[1], val_dataset[1], predict_dataset[1]))
            dna_data = np.vstack((train_dataset[2], val_dataset[2], predict_dataset[2]))

            labels_train = np.load(os.path.join(data_dir, '3types', 'training_3types.npy'))
            labels_val = np.load(os.path.join(data_dir, '3types', 'validation_3types.npy'))
            labels_predict = np.load(os.path.join(data_dir, '3types', 'predict_3types.npy'))
            cancer_types_samples = np.concatenate((labels_train, labels_val, labels_predict))

            full_dataset = datasets.TCGADataset(rna_data, gcn_data, dna_data)

            # 1 batch for all data
            data_loader = torch.utils.data.DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)

            title = "{} of Experts 3 Cancer Types (BRCA, KIRC, LUAD) Latent Space: \n" \
                    "Epochs : {}, Batch size : {}, Latent space : {}, LR: {}"\
                .format("Mixture" if model.use_mixture else "Product", checkpoint['epochs'], checkpoint['batch_size'], model.latent_dim, checkpoint['lr'])
            save_dir = "/Users/bram/Desktop/followup_UMAP"
            save_file = "{}/UMAP {} 3 Cancer Types E{} Dim{} LR{}.png"\
                .format(save_dir, "Mixture" if model.use_mixture else "Product", checkpoint['epochs'], model.latent_dim, checkpoint['lr'])
            background = "black"
            color_key_cmap = "Paired"

        else:
            tcga_data = datasets.TCGAData(indices_path=results_dir)
            cancer_types_samples = np.load(os.path.join(data_dir, 'shuffle_cancertype_labels.npy'))

            full_dataset = tcga_data.get_data_partition("all")

            # 1 batch for all data
            data_loader = torch.utils.data.DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)

            title = "{} of Experts Full TCGA Data Latent Space: \n" \
                    "Epochs : {}, Batch size : {}, Latent space : {}, LR: {}"\
                .format("Mixture" if model.use_mixture else "Product", checkpoint['epochs'], checkpoint['batch_size'], model.latent_dim, checkpoint['lr'])
            save_dir = "/Users/bram/Desktop/followup_UMAP"
            save_file = "{}/UMAP {} All Data E{} Dim{} LR{}.png"\
                .format(save_dir, "Mixture" if model.use_mixture else "Product", checkpoint['epochs'], model.latent_dim, checkpoint['lr'])
            background = "white"
            color_key_cmap = "Spectral"

        # Get Latent Space
        print("Entering Data into model, fetching Z and fitting UMAP.")
        z = 0
        for batch_idx, (rna, gcn, dna) in enumerate(data_loader):
            if model.use_mixture:
                mu, logvar = model.get_mixture_params(rna=rna, gcn=gcn, dna=dna)
            else:
                mu, logvar = model.get_product_params(rna=rna, gcn=gcn, dna=dna)

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

        p = umap.plot.points(mapper, labels=cancer_types_samples, color_key_cmap=color_key_cmap, background=background)
        umap.plot.plt.title(title)
        # umap.plot.plt.legend()
        umap.plot.plt.savefig(save_file, dpi=1600)
        umap.plot.plt.show()


if __name__ == "__main__":
    umap_plotter = UMAP(cancer3=False)
    umap_plotter.plot_umap()
