"""
Python Script for UMAP plot of the Z from MOFA+ TCGA's dataset
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import umap
import umap.plot

Z = pd.read_csv("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_Z.csv")

# Get Z matrix
unique_factors = np.unique(Z['factor'].values)

z_matrix = []

for factor in tqdm(range(len(unique_factors))):
    z_matrix.append((Z['value'].loc[Z['factor'] == unique_factors[factor]]).values)

Z = np.matrix(z_matrix).transpose()

cancer_types_samples = np.load("/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_cancertypes_samples.npy")
trained_indices = np.load("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_MOFA_DATA_indices.npy")
cancer_types_samples = cancer_types_samples[trained_indices]

mapper = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean'
).fit(Z)

print("Umap has been fit. Now plot")

p = umap.plot.points(mapper, labels=cancer_types_samples)
umap.plot.plt.title(
    "MOFA+ Latent Space UMAP : Colored per cancer type: \n10 Factors, 3 views, 1 group")
umap.plot.plt.legend()
umap.plot.plt.show()