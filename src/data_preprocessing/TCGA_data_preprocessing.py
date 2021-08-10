"""
Python Script for processing TCGA's dataset:
gene expression RNAseq - Batch effects normalized mRNA data

1) Read the raw .xena files from the TCGA website
2) Replace NaN values with 0
3) Normalize data
4) NOTE: For MoE should be normalized between 0.00000001 and 0.99999999
5) Outputs only the 3,000 most variable genes based on Mean Absolute Deviation (MAD)
6) Converts the .xena file format to .csv

Sources are pulled straight from:
https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
RNA-seq: Batch effects normalized mRNA data
Gene Copy Number: gene-level copy number (gistic2)
DNA Methylation: DNA methylation (Methylation450K)
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------- SETUP INPUT/OUTPUT PATHS ----------

# input
rna = "/Users/bram/Downloads/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
dna = "/Users/bram/Downloads/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz"
# output
rna_out = "/Users/bram/Downloads/RNASeq_5000MAD.csv"
dna_out = "/Users/bram/Downloads/DNAMe_5000MAD.csv"

NUM_MAD_FEATURES = 5000
# NOTE: For large data files, this code might have to run in steps (per dataset)

######################################################

print("Reading in .xena data files")
print("Preselecting {} MAD columns for DNA since file is too large".format(NUM_MAD_FEATURES))

# DNA File too large, so preselect MAD columns
# The following loop assumes that features are on the rows and samples on the columns
chunksize = 2000  # Arbitrary, but low memory usage
feature_mad_values = []
for chunk in tqdm(pd.read_table(dna, chunksize=chunksize, index_col=0)):
    chunk = chunk.fillna(0)  # Make sure NaN is not included in calculation
    mad_genes_chunk = chunk.mad(axis=1)  # Get the MAD values of the chunk
    feature_mad_values.extend(mad_genes_chunk.values)  # Add them to a list

# We have the MAD values of every row (feature) in feature_mad_values
# Argsort but in descending order (highest first)
highest_mad_feature_indices = np.argsort(np.array(feature_mad_values))[::-1]

# Skip rows (features) that were not in the highest 3000
skiprows = np.setdiff1d(np.arange(1, len(highest_mad_feature_indices) + 1), highest_mad_feature_indices[:NUM_MAD_FEATURES] + 1)  # Highest MAD values but indexed from 1

RNA_DATA = pd.read_table(rna, index_col=0)
print("Done reading RNA-seq")
# Now DNA_DATA is better to work with
DNA_DATA = pd.read_table(dna, index_col=0, skiprows=skiprows)
print("Done reading DNA Methylation")

# Replace NaN values with 0
RNA_DATA = RNA_DATA.fillna(0)
DNA_DATA = DNA_DATA.fillna(0)

# Make sure datasets have the same ordering (samples in rows and features as columns)
RNA_DATA = RNA_DATA.transpose()
DNA_DATA = DNA_DATA.transpose()

# Find common samples between datasets
common_samples = np.intersect1d(DNA_DATA.index.values, RNA_DATA.index.values)
print("Nr of common samples", len(common_samples))

# Only take the common samples
RNA_DATA = RNA_DATA.loc[common_samples]
DNA_DATA = DNA_DATA.loc[common_samples]

# Determine most variably expressed genes and subset
# From tybalt/process_data.ipynb
# https://github.com/greenelab/tybalt/blob/master/process_data.ipynb

mad_genes_rna = RNA_DATA.mad(axis=0).sort_values(ascending=False)
top_mad_genes_rna = mad_genes_rna.iloc[0:NUM_MAD_FEATURES, ].index
RNA_DATA = RNA_DATA.loc[:, top_mad_genes_rna]

print("RNA MAD Shape", RNA_DATA.shape)

mad_genes_dna = DNA_DATA.mad(axis=0).sort_values(ascending=False)
top_mad_genes_dna = mad_genes_dna.iloc[0:NUM_MAD_FEATURES, ].index
DNA_DATA = DNA_DATA.loc[:, top_mad_genes_dna]

print("DNA MAD Shape", DNA_DATA.shape)

# Either normalize data by dividing all values by largest number
# OR for MoE repository: Values should be normalized between 0.00000001 and 0.99999999
# Comment out what suits your needs

# # Normal normalization (between -1 and 1)
#
# # Divide by either highest positive or negative value (abs)
# if abs(np.min(RNA_DATA.values)) <= 1 and abs(np.max(RNA_DATA.values)) <= 1:  # Already normalized
#     print("Skipping normalization of RNA-seq data :  Already Normalized.")
# else:
#     RNA_DATA = RNA_DATA / abs(np.max(RNA_DATA.values)) if (abs(np.max(RNA_DATA.values)) > abs(np.min(RNA_DATA.values))) else RNA_DATA / abs(np.min(RNA_DATA.values))
#
# if abs(np.min(DNA_DATA.values)) <= 1 and abs(np.max(DNA_DATA.values)) <= 1:  # Already normalized
#     print("Skipping normalization of DNA Methylation data :  Already Normalized.")
# else:
#     DNA_DATA = DNA_DATA / abs(np.max(DNA_DATA.values)) if (abs(np.max(DNA_DATA.values)) > abs(np.min(DNA_DATA.values))) else DNA_DATA / abs(np.min(DNA_DATA.values))
#
# assert np.max(RNA_DATA.values) <= 1 and np.max(DNA_DATA.values) <= 1, "Not correctly normalized (value exceeds 1)"
# assert np.min(RNA_DATA.values) >= -1 and np.min(DNA_DATA.values) >= -1, "Not correctly normalized (value below -1)"

# Mixture of Experts repository positive normalization (between epsilon and 1 - epsilon)
EPSILON = 1e-8

if abs(np.min(RNA_DATA.values)) < 1 and abs(np.max(RNA_DATA.values)) < 1:  # Already normalized
    print("Skipping normalization of RNA-seq data :  Already Normalized.")
else:
    RNA_DATA = RNA_DATA + abs(np.min(RNA_DATA.values))  # Transform to all positive values
    RNA_DATA = RNA_DATA / np.max(RNA_DATA.values)  # Normalize
    RNA_DATA = RNA_DATA.clip(EPSILON, 1 - EPSILON)  # Clamp to 1e-8 and (1 - 1e-8)

if abs(np.min(DNA_DATA.values)) < 1 and abs(np.max(DNA_DATA.values)) < 1:  # Already normalized
    print("Skipping normalization of DNA Methylation data :  Already Normalized.")
else:
    DNA_DATA = DNA_DATA + abs(np.min(DNA_DATA.values))
    DNA_DATA = DNA_DATA / np.max(DNA_DATA.values)
    DNA_DATA = DNA_DATA.clip(EPSILON, 1 - EPSILON)


assert np.max(RNA_DATA.values) < 1 and np.max(DNA_DATA.values) < 1, "Not correctly normalized for MoE (value exceeds (1 - 1e-8))"
assert np.min(RNA_DATA.values) >= EPSILON and np.min(DNA_DATA.values) >= EPSILON, "Not correctly normalized for MoE (value below 1e-8)"

# Check if shapes are still equal and ready to be processed for MAD features (same number of samples)
assert RNA_DATA.shape[0] == DNA_DATA.shape[0], "Data not of same shape after normalization (rows)"

# Check if each dataset now has NUM_MAD_FEATURES columns
assert RNA_DATA.shape[1] <= NUM_MAD_FEATURES and DNA_DATA.shape[1] <= NUM_MAD_FEATURES, \
       "Data not of same shape after normalization (columns)"

# Write most variable genes to .csv file
print("Now writing preprocessed files to .csv format")
RNA_DATA.to_csv(rna_out)
DNA_DATA.to_csv(dna_out)

print(RNA_DATA)
print(DNA_DATA)

print("Done writing! Printing resulting datasets and exiting program.")
