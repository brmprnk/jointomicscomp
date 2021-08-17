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
    omic_data1 = np.load(args['data_path1'])
    omic_data2 = np.load(args['data_path2'])
    sample_names = np.load(args['sample_names'])
    cancer_type_index = np.load(args['cancer_type_index'])
    cancertypes = np.load(args['cancertypes'])

    logger.success("Finished reading in data : shape {}".format(omic_data1.shape))

    assert len(omic_data1) == len(omic_data2), "Modalities do not have the same number of samples, exiting program."

    print(omic_data1)
    print(sample_names)
    print(cancer_type_index)
    print(cancertypes)
    # Create lists for each column in the final dataset
    SAMPLE_DATA = []
    FEATURE_DATA = []
    VALUE_DATA = []
    VIEW_DATA = np.full((args['num_features'] * omic_data1.shape[0], ), "GE").tolist()  # We will have 5000 features for each sample in this view

    # For each sample, add an entry for each feature to the columns of the output Dataframe
    # Index represents sample name
    index = 0
    for sample in tqdm(omic_data1, total=omic_data1.shape[0], unit='samples '):

        # Add Sample Name to SAMPLE_DATA
        sample_list = [sample_names[index]] * args['num_features']
        SAMPLE_DATA.extend(sample_list)

        # Add all features to FEATURE_DATA
        # feature_list = [cancertypes[cancer_type_index[index]]] * args['num_features']
        FEATURE_DATA.extend(np.arange(5000))

        # Add all values to VALUE_DATA
        VALUE_DATA.extend(sample)
        index += 1

    # Add View Name to VIEW_DATA
    VIEW_DATA.extend(np.full((args['num_features'] * omic_data2.shape[0], ), "ME").tolist())

    # For each sample, add an entry for each feature to the columns of the output Dataframe
    index = 0
    for sample in tqdm(omic_data2, total=omic_data2.shape[0], unit='samples '):

        # Add Sample Name to SAMPLE_DATA
        sample_list = [sample_names[index]] * args['num_features']
        SAMPLE_DATA.extend(sample_list)

        # Add all features to FEATURE_DATA
        # feature_list = [cancertypes[cancer_type_index[index]]] * args['num_features']
        FEATURE_DATA.extend(np.arange(5000))

        # Add all values to VALUE_DATA
        VALUE_DATA.extend(sample)
        index += 1

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
    mofa_data_frame.to_csv(os.path.join(output_data_path, 'mofa_GE_ME_5000MAD.csv'), index=False)
    logger.success("Done writing. Exiting Program.")
