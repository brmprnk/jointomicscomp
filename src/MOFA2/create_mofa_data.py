"""
Python Script for processing q dataset.
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

GROUP is an optional column, and will be omitted in this case
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.util import logger


def create_mofa_dataframe(args: dict) -> None:
    """

    @param args: Dictionary containing input parameters
    @return:
    """
    logger.info("Preprocessing data for use in MOFA+")

    # Save to same dir where data is retrieved from
    output_data_path = os.path.dirname(args['data_path1'])

    # Use read_table instead if using .xena file
    logger.info("Reading in data from all modalities...")
    omic_data1 = pd.read_csv(args['data_path1'], index_col=0)
    omic_data2 = pd.read_csv(args['data_path2'], index_col=0)
    logger.success("Finished reading in data")

    assert len(omic_data1.index.values) == len(omic_data2.index.values), "Modalities do not have the same number of samples, exiting program."

    # Create lists for each column in the final dataset
    SAMPLE_DATA = []
    FEATURE_DATA = []
    VALUE_DATA = []
    VIEW_DATA = np.full((args['num_features'] * omic_data1.shape[0], ), "RNA-seq").tolist()  # We will have 5000 features for each sample in this view

    # For each sample, add an entry for each feature to the columns of the output Dataframe
    # Index represents sample name
    for index, row in tqdm(omic_data1.iterrows(), total=omic_data1.shape[0], unit='samples '):

        # Sample row
        sample = omic_data1.loc[index]

        # Add Sample Name to SAMPLE_DATA
        sample_list = [index] * args['num_features']
        SAMPLE_DATA.extend(sample_list)

        # Add all features to FEATURE_DATA
        FEATURE_DATA.extend(sample.index.values.tolist())

        # Add all values to VALUE_DATA
        VALUE_DATA.extend(sample.values.tolist())

    # Add View Name to VIEW_DATA
    VIEW_DATA.extend(np.full((args['num_features'] * omic_data2.shape[0], ), "DNA").tolist())

    # For each sample, add an entry for each feature to the columns of the output Dataframe
    for index, row in tqdm(omic_data2.iterrows(), total=omic_data1.shape[0], unit='samples '):

        sample = omic_data2.loc[index]

        # Add Sample Name to SAMPLE_DATA
        sample_list = [index] * args['num_features']
        SAMPLE_DATA.extend(sample_list)

        # Add all features to FEATURE_DATA
        FEATURE_DATA.extend(sample.index.values.tolist())

        # Add all values to VALUE_DATA
        VALUE_DATA.extend(sample.values.tolist())

    logger.info("Creating pandas DataFrame of the data...")

    # Create DataFrame
    mofa_data_frame = pd.DataFrame(data={
        "sample": SAMPLE_DATA,
        "feature": FEATURE_DATA,
        "view": VIEW_DATA,
        "value": VALUE_DATA
    })

    logger.success("Dataframe created. Final MOFA DATA Shape : {}".format(mofa_data_frame.shape))
    logger.info("Writing MOFA+ data to : {}".format(output_data_path))
    mofa_data_frame.to_csv(os.path.join(output_data_path, 'mofa_rnaseq_dname_5000MAD.csv'), index=False)
    logger.success("Done writing. Exiting Program.")
