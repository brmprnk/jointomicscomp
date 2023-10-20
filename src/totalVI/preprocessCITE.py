import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import muon
import scanpy as sc
from scvi.data import pbmc_seurat_v4_cite_seq
from scvi.model import TOTALVI
import sys
import numpy as np
import pickle
from copy import deepcopy

def loadCITE(save=False, dataPrefix=None):

    if dataPrefix is None:
        dataPrefix = '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/'

    dataLoc = dataPrefix + 'data/scvi-cite/'

    # load data using default filters
    adata = pbmc_seurat_v4_cite_seq(save_path=dataLoc, apply_filters=True, aggregate_proteins=True, mask_protein_batches=0)

    adata.layers["counts"] = adata.X.copy()
    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    adata.obs_names_make_unique()

    #adata.obs['batch'] = adata.obs['donor'].astype(str) + '-' + adata.obs['time'].astype(str)


    protein_adata = ad.AnnData(adata.obsm["protein_counts"])
    protein_adata.obs_names = adata.obs_names
    del adata.obsm["protein_counts"]
    mdata = md.MuData({"rna": adata, "protein": protein_adata})
    mdata.var_names_make_unique()

    sc.pp.highly_variable_genes(
        mdata.mod["rna"],
        n_top_genes=5000,
        flavor="seurat_v3",
    #    batch_key="batch",
        layer="counts")

    mdata.mod["rna_subset"] = mdata.mod["rna"][:, mdata.mod["rna"].var["highly_variable"]].copy()
    mdata.update()

    if save:
        testInd = np.where(mdata.obs['rna:donor'] == 'P1')[0]
        validInd = np.where(mdata.obs['rna:donor'] == 'P2')[0]
        trainInd = np.where(np.logical_not(np.logical_or((mdata.obs['rna:donor'] == 'P1'), ( mdata.obs['rna:donor'] == 'P2'))))[0]

        print(trainInd.shape)
        print(validInd.shape)
        print(testInd.shape)

        np.save(dataLoc + 'trainInd', trainInd)
        np.save(dataLoc + 'validInd', validInd)
        np.save(dataLoc + 'testInd', testInd)

        np.save(dataLoc + 'RNA.npy', mdata['rna_subset'].layers['counts'].todense())
        np.save(dataLoc + 'ADT.npy', mdata.mod['protein'].X)

        np.save(dataLoc + 'sampleNames.npy', np.array(mdata.obs.index))

        cellTypesL2 = np.array(sorted(mdata.obs['rna:celltype.l2'].unique()))
        ct2index = dict()
        for i,c in enumerate(cellTypesL2):
            ct2index[c] = i

        y = np.array(mdata.obs['rna:celltype.l2'].map(ct2index))

        np.save(dataLoc + 'celltypes.npy', cellTypesL2)
        np.save(dataLoc + 'celltype.npy', y)

        cellTypesL3 = np.array(sorted(mdata.obs['rna:celltype.l3'].unique()))

        ct2index = dict()
        for i,c in enumerate(cellTypesL3):
            ct2index[c] = i

        y = np.array(mdata.obs['rna:celltype.l3'].map(ct2index))

        np.save(dataLoc + 'celltypes_l3.npy', cellTypesL3)

        np.save(dataLoc + 'celltype_l3.npy', y)

        np.save(dataLoc + 'featureNamesADT.npy', mdata['protein'].var_names)
        np.save(dataLoc + 'featureNamesRNA.npy', mdata['rna_subset'].var_names)


    mdataTest = mdata[mdata.obs[mdata.obs['rna:donor'] == 'P1'].index].copy()
    mdataTest.update()

    mdataTrain = mdata[mdata.obs[mdata.obs['rna:donor'] != 'P1'].index].copy()
    mdataTrain.update()

    ii = np.where(mdataTrain.obs['rna:donor'] == 'P2')[0]
    ii1 = np.where(mdataTrain.obs['rna:donor'] != 'P2')[0]
    ind = np.hstack([ii1, ii])

    mdataTrain['rna'].X = mdataTrain['rna'].X[ind]
    mdataTrain['rna'].layers['counts'] = mdataTrain['rna'].X.copy()
    mdataTrain['rna'].obs = mdataTrain['rna'].obs.iloc[ind]

    mdataTrain['rna_subset'].X = mdataTrain['rna_subset'].X[ind]
    mdataTrain['rna_subset'].layers['counts'] = mdataTrain['rna_subset'].X.copy()
    mdataTrain['rna_subset'].obs = mdataTrain['rna_subset'].obs.iloc[ind]

    mdataTrain['protein'].X = mdataTrain['protein'].X[ind]
    mdataTrain['protein'].obs = mdataTrain['protein'].obs.iloc[ind]

    mdataTrain.update()

    # TOTALVI.setup_mudata(
    #     mdata,
    #     rna_layer="counts",
    #     protein_layer=None,
    #     #batch_key="batch",
    #     modalities={
    #         "rna_layer": "rna_subset",
    #         "protein_layer": "protein",
    #         #"batch_key": "rna_subset",
    #     }
    # )
    #

    TOTALVI.setup_mudata(
        mdataTrain,
        rna_layer="counts",
        protein_layer=None,
        #batch_key="batch",
        modalities={
            "rna_layer": "rna_subset",
            "protein_layer": "protein",
            #"batch_key": "rna_subset",
        }
    )


    TOTALVI.setup_mudata(
        mdataTest,
        rna_layer="counts",
        protein_layer=None,
        #batch_key="batch",
        modalities={
            "rna_layer": "rna_subset",
            "protein_layer": "protein",
            #"batch_key": "rna_subset",
        }
    )


    return mdataTrain, mdataTest


def maskProteins(mudata):
    mydata = deepcopy(mudata)

    mydata['protein'].X = np.zeros_like(mydata['protein'].X)

    TOTALVI.setup_mudata(
        mydata,
        rna_layer="counts",
        protein_layer=None,
        #batch_key="batch",
        modalities={
            "rna_layer": "rna_subset",
            "protein_layer": "protein",
            #"batch_key": "rna_subset",
        }
    )


    return mydata

def maskRNA(mudata):
    mydata = deepcopy(mudata)

    mydata['rna'].X = mydata['rna'].layers['counts'] - mydata['rna'].X
    mydata['rna'].layers['counts'] = mydata['rna'].X.copy()

    mydata['rna_subset'].X = mydata['rna_subset'].layers['counts'] - mydata['rna_subset'].X
    mydata['rna_subset'].layers['counts'] = mydata['rna_subset'].X.copy()

    TOTALVI.setup_mudata(
        mydata,
        rna_layer="counts",
        protein_layer=None,
        #batch_key="batch",
        modalities={
            "rna_layer": "rna_subset",
            "protein_layer": "protein",
            #"batch_key": "rna_subset",
        }
    )


    return mydata
