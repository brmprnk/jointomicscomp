"""
Python Script for calculating the prediction loss from MOFA+ TCGA's dataset
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np

rna_path = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_RNA_3000MAD_cancertypeknown.csv"
gcn_path = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_GCN_3000MAD_cancertypeknown.csv"
dna_path = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_DNA_3000MAD_cancertypeknown.csv"

result_output_path = "/Users/bram/rp-group-21-bpronk/results/80split_recon_loss.npy"

print("Reading original data...")
RNA_DATA = pd.read_csv(rna_path, index_col=0)
GCN_DATA = pd.read_csv(gcn_path, index_col=0)
DNA_DATA = pd.read_csv(dna_path, index_col=0)

trained_indices = np.load("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_MOFA_DATA_indices.npy")
RNA_DATA = RNA_DATA.iloc[trained_indices]
GCN_DATA = GCN_DATA.iloc[trained_indices]
DNA_DATA = DNA_DATA.iloc[trained_indices]

print(RNA_DATA.shape)
print(GCN_DATA.shape)
print(DNA_DATA.shape)
print("Finished reading original data...")

# Now get the results from MOFA
# W = 3000 x 10 : Factors on columns, Features on Rows
# Z = 10 * 9992 : Samples on columns, Factors on rows

# Y = WZ = 3000 * 9992 : Features on rows, Samples on columns

W = pd.read_csv("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_W.csv")
Z = pd.read_csv("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_Z.csv")

# Get Z matrix
unique_factors = np.unique(Z['factor'].values)

z_matrix = []

for factor in tqdm(range(len(unique_factors))):
    z_matrix.append((Z['value'].loc[Z['factor'] == unique_factors[factor]]).values)

Z = np.matrix(z_matrix)

# Get W matrix for each modality
W_RNA = W.loc[W['view'] == "RNA-seq"]
W_GCN = W.loc[W['view'] == "GENE CN"]
W_DNA = W.loc[W['view'] == "DNA"]

unique_features_rna = W_RNA['feature'].values[::10]
matrix = []
for i in tqdm(range(unique_features_rna.shape[0])):
    matrix.append(W_RNA['value'].loc[W_RNA['feature'] == unique_features_rna[i]])

W_RNA = np.matrix(matrix)

unique_features_gcn = W_GCN['feature'].values[::10]
matrix = []
for i in tqdm(range(unique_features_gcn.shape[0])):
    matrix.append(W_GCN['value'].loc[W_GCN['feature'] == unique_features_gcn[i]])

W_GCN = np.matrix(matrix)

unique_features_dna = W_DNA['feature'].values[::10]
matrix = []
for i in tqdm(range(unique_features_dna.shape[0])):
    matrix.append(W_DNA['value'].loc[W_DNA['feature'] == unique_features_dna[i]])

W_DNA = np.matrix(matrix)

print(W_RNA.shape, Z.shape)
print(W_GCN.shape, Z.shape)
print(W_DNA.shape, Z.shape)



# Now get original values back (Y = WZ)
Y_RNA = np.matmul(W_RNA, Z)
Y_GCN = np.matmul(W_GCN, Z)
Y_DNA = np.matmul(W_DNA, Z)

# Get back in the form of original data
Y_RNA = Y_RNA.transpose()
Y_GCN = Y_GCN.transpose()
Y_DNA = Y_DNA.transpose()

print(Y_RNA)
print(Y_RNA.shape)
print(Y_GCN)
print(Y_GCN.shape)
print(Y_DNA)
print(Y_DNA.shape)

# input, predict
rna_recon_loss = mean_squared_error(RNA_DATA.values, Y_RNA)
print("RNA Reconstruction loss = ", rna_recon_loss)

# input, predict
gcn_recon_loss = mean_squared_error(GCN_DATA.values, Y_GCN)
print("GCN Reconstruction loss = ", gcn_recon_loss)

# input, predict
dna_recon_loss = mean_squared_error(DNA_DATA.values, Y_GCN)
print("DNA Reconstruction loss = ", dna_recon_loss)

result = np.array([rna_recon_loss, gcn_recon_loss, dna_recon_loss])
np.save(result_output_path, result)
