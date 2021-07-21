"""
Python Script for processing TCGA's dataset:
gene expression RNAseq - Batch effects normalized mRNA data
For use with Multi-Omics Factor Analysis V2 (MOFA+) https://biofam.github.io/MOFA2/

MOFA+ requires a data set of the following format: 
A DataFrame with columns ["sample","feature","view","group","value"]
Example:

sample   group   feature    value   view
-----------------------------------------------
sample1  groupA  gene1      2.8044  RNA
sample1  groupA  gene3      2.2069  RNA
sample2  groupB  gene2      0.1454  RNA
sample2  groupB  gene1      2.7021  RNA
sample2  groupB  promoter1  3.8618  Methylation
sample3  groupB  promoter2  3.2545  Methylation
sample3  groupB  promoter3  1.5014  Methylation

GROUP is an optional column, and will be omitted in my case
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

############## SETUP INPUT/OUTPUT PATHS ##############

# 3000 most variable subjects
NUM_FEATURES = 3000
rna_path = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_RNA_3000MAD_cancertypeknown.csv"
gcn_path = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_GCN_3000MAD_cancertypeknown.csv"
dna_path = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_DNA_3000MAD_cancertypeknown.csv"

output_data_path = "/Users/bram/rp-group-21-bpronk/data/80split_shuffle_MOFA_DATA.csv"

######################################################

# Use read_table instead if using .xena file
print("Reading in data from all modalities...")
RNA_DATA = pd.read_csv(rna_path, index_col=0)
GENE_DATA = pd.read_csv(gcn_path, index_col=0)
DNA_DATA = pd.read_csv(dna_path, index_col=0)
print("Finished reading in data")

# MVAE Model uses 70-10-20 split.
# For fair comparison, MOFA+ will also omit 20% of data, but does not account for the MVAE validation split, since that is an advantage of MOFA+
assert len(RNA_DATA.index.values) == len(GENE_DATA.index.values) == len(DNA_DATA.index.values), "Modalities do not have the same number of samples, exiting program."

total_samples = len(RNA_DATA.index.values)
training_samples = int(total_samples * 0.8)

# Select random samples by index, replace=False so samples can not be selected more than once
random_indices = np.random.choice(a=total_samples, size=training_samples, replace=False)
np.save("/Users/bram/rp-group-21-bpronk/data/80split_shuffle_MOFA_DATA_indices.npy", random_indices)
RNA_DATA = RNA_DATA.iloc[random_indices]
GENE_DATA = GENE_DATA.iloc[random_indices]
DNA_DATA = DNA_DATA.iloc[random_indices]

print("MOFA+ input (samples, features)", RNA_DATA.shape, GENE_DATA.shape, DNA_DATA.shape)

# Create lists for each column in the final dataset
SAMPLE_DATA = []
FEATURE_DATA = []
VALUE_DATA = []
VIEW_DATA = np.full((NUM_FEATURES * RNA_DATA.shape[0], ), "RNA-seq").tolist() # We will have 5000 features for each sample in this view

# For each sample, add an entry for each feature to the columns of the output Dataframe
# Index represents sample name
for index, row in tqdm(RNA_DATA.iterrows(), total=RNA_DATA.shape[0], unit='samples '):

    # Sample row
    sample = RNA_DATA.loc[index]

    # Add Sample Name to SAMPLE_DATA
    sample_list = [index] * NUM_FEATURES
    SAMPLE_DATA.extend(sample_list)

    # Add all features to FEATURE_DATA
    FEATURE_DATA.extend(sample.index.values.tolist())

    # Add all values to VALUE_DATA
    VALUE_DATA.extend(sample.values.tolist())

# Add View Name to VIEW_DATA
VIEW_DATA.extend(np.full((NUM_FEATURES * GENE_DATA.shape[0], ), "GENE CN").tolist())

# For each sample, add an entry for each feature to the columns of the output Dataframe
for index, row in tqdm(GENE_DATA.iterrows(), total=RNA_DATA.shape[0], unit='samples '):

    sample = GENE_DATA.loc[index]

    # Add Sample Name to SAMPLE_DATA
    sample_list = [index] * NUM_FEATURES
    SAMPLE_DATA.extend(sample_list)

    # Add all features to FEATURE_DATA
    FEATURE_DATA.extend(sample.index.values.tolist())

    # Add all values to VALUE_DATA
    VALUE_DATA.extend(sample.values.tolist())

# Add View Name to VIEW_DATA
VIEW_DATA.extend(np.full((NUM_FEATURES * DNA_DATA.shape[0], ), "DNA").tolist())

# For each sample, add an entry for each feature to the columns of the output Dataframe
for index, row in tqdm(DNA_DATA.iterrows(), total=RNA_DATA.shape[0], unit='samples '):

    sample = DNA_DATA.loc[index]

    # Add Sample Name to SAMPLE_DATA
    sample_list = [index] * NUM_FEATURES
    SAMPLE_DATA.extend(sample_list)

    # Add all features to FEATURE_DATA
    FEATURE_DATA.extend(sample.index.values.tolist())

    # Add all values to VALUE_DATA
    VALUE_DATA.extend(sample.values.tolist())

# Create DataFrame
mofa_data_frame = pd.DataFrame(data={
    "sample": SAMPLE_DATA,
    "feature": FEATURE_DATA,
    "view" : VIEW_DATA,
    "value": VALUE_DATA
})

print("Final MOFA DATA Shape", mofa_data_frame.shape)
print("Writing to", output_data_path)
mofa_data_frame.to_csv(output_data_path, index=False)
print("Done writing. Exiting Program.")
