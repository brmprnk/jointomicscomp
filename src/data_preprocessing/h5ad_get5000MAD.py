"""
Reading in the h5ad file is so memory expensive, that further operations on it
are prohibited, even when using multiprocesses and freeing memory after each operation.
Make sure to either:
    > Have more than 16GB memory
    > Run the file in increments
"""
import scanpy as sc
import numpy as np
import os
from pathlib import Path

rna_path = "/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_RNA.h5ad"
adt_path = "/home/bram/jointomicscomp/data/CELL/pbmc_multimodal_ADT.h5ad"

rna_save_dir = os.path.dirname(rna_path) + '/'
adt_save_dir = os.path.dirname(adt_path) + '/'

# First process RNA, then protein
adata = sc.read_h5ad(rna_path)

sc.pp.highly_variable_genes(adata, n_top_genes=5000)

adata.obs.to_csv(rna_save_dir + Path(rna_path).stem + "_5000MAD_obs.csv")
adata.var.to_csv(rna_save_dir + Path(rna_path).stem + "_5000MAD_var.csv")

np.save(rna_save_dir + Path(rna_path).stem + "_5000MAD.npy", adata.X[:, adata.var['highly_variable']].astype(np.float64))

np.save(rna_save_dir + Path(rna_path).stem + "_5000MAD_featureNames.npy", adata[:, adata.var['highly_variable']].var['features'].values)
np.save(rna_save_dir + Path(rna_path).stem + "_5000MAD_sampleNames.npy", adata[:, adata.var['highly_variable']].obs.index.values)

celltype_l2 = adata[:, adata.var['highly_variable']].obs['celltype.l2'].values
celltype_l2_unique = np.unique(celltype_l2)

for i in range(len(celltype_l2)):
    celltype_l2[i] = np.where(celltype_l2_unique == celltype_l2[i])[0]

np.save(rna_save_dir + Path(rna_path).stem + "_5000MAD_cellType.npy", celltype_l2.astype(int))
np.save(rna_save_dir + Path(rna_path).stem + "_5000MAD_cellTypes.npy", celltype_l2_unique.astype(str))


# Now do protein
adata = sc.read_h5ad(adt_path)

sc.pp.highly_variable_genes(adata, n_top_genes=5000)

adata.obs.to_csv(adt_save_dir + Path(adt_path).stem + "_5000MAD_obs.csv")
adata.var.to_csv(adt_save_dir + Path(adt_path).stem + "_5000MAD_var.csv")

np.save(adt_save_dir + Path(adt_path).stem + "_5000MAD.npy", adata.X[:, adata.var['highly_variable']].astype(np.float64))

np.save(adt_save_dir + Path(adt_path).stem + "_5000MAD_featureNames.npy", adata[:, adata.var['highly_variable']].var['features'].values)
np.save(adt_save_dir + Path(adt_path).stem + "_5000MAD_sampleNames.npy", adata[:, adata.var['highly_variable']].obs.index.values)

celltype_l2 = adata[:, adata.var['highly_variable']].obs['celltype.l2'].values
celltype_l2_unique = np.unique(celltype_l2)

for i in range(len(celltype_l2)):
    celltype_l2[i] = np.where(celltype_l2_unique == celltype_l2[i])[0][0]

np.save(adt_save_dir + Path(adt_path).stem + "_5000MAD_cellType.npy", celltype_l2.astype(int))
np.save(adt_save_dir + Path(adt_path).stem + "_5000MAD_cellTypes.npy", celltype_l2_unique.astype(str))
