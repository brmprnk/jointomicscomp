from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import datasets

pca_rna = PCA(n_components=128)
pca_gcn = PCA(n_components=128)
pca_dna = PCA(n_components=128)

indices_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE shuffle lr 1e-4 latent128"
tcga_data = datasets.TCGAData(indices_path=indices_path)

train_dataset = tcga_data.get_data_partition("train")
val_dataset = tcga_data.get_data_partition("val")
predict_dataset = tcga_data.get_data_partition("predict")

pca_rna.fit(train_dataset.rna_data)
pca_gcn.fit(train_dataset.gcn_data)
pca_dna.fit(train_dataset.dna_data)

samples = len(val_dataset.rna_data)

z_space_rna = pca_rna.transform(val_dataset.rna_data)
z_space_gcn = pca_rna.transform(val_dataset.gcn_data)
z_space_dna = pca_rna.transform(val_dataset.dna_data)

reconstructed_rna = pca_rna.inverse_transform(z_space_rna)
reconstructed_gcn = pca_gcn.inverse_transform(z_space_gcn)
reconstructed_dna = pca_dna.inverse_transform(z_space_dna)

mse_rna = mean_squared_error(val_dataset.rna_data, reconstructed_rna)
mse_gcn = mean_squared_error(val_dataset.gcn_data, reconstructed_gcn)
mse_dna = mean_squared_error(val_dataset.dna_data, reconstructed_dna)

print("PCA loss RNA |||", mse_rna)
print("PCA loss GCN |||", mse_gcn)
print("PCA loss DNA |||", mse_dna)
