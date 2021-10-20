# jointomicscomp              <img src="https://www.abuse.nl/assets/logos/tudelft.png" width="100" height="50">

[![Python Version](https://img.shields.io/static/v1.svg?label=Python%20Version&message=3.9&color=blue)](https://www.python.org/downloads)
[![Conda Install](https://anaconda.org/conda-forge/terraform-provider-github/badges/installer/conda.svg)](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

Wrapper and implementation for comparing five models for multiomics data integration; MOFA+, Mixture-of-Experts, Product-of-Experts, Multi-view Info Bottleneck, and CGAE models.

## Installation
<!---
This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.
--->
Recommended installation uses a new Anaconda environment. To ease the process, this project includes an environment file.
This can be plugged into Anaconda [following this short tutorial](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Then to run the project, simply call the ```run.py``` wrapper with the desired config file like so:
```bash
python run.py -c configs/geme.yaml
```
This default call to run.py will run all implemented models. For specific models, use one or combine the 
```-poe -moe -mofa -mvib -cgae``` flags.
To see additional arguments, use argparse's built in -h command:
```bash
python run.py -h
```


- The experiment name is set in the config file, or in the command line using ```-e experiment_name```
- All results and model outputs will be stored in the project's ```results``` directory
- All metrics are written to a TensorBoard file, also found in the results folder
- All informational print statements are saved to a log.txt file, using a logger found in the util folder.


## To-do's
To see some open issues, have a look at this [Drive doc](https://docs.google.com/document/d/1R6pXmTQIgCXdm_zTyRQ4GYI5oybIU4Vas66auJLRIJk/edit?usp=sharing)

## General Status
- For every model, the Z is saved to the results folder, together with a UMAP representation.
- Imputation is done right after model training. The code for the imputation file is therefore found in the main file of each model.
- All options for data/model specific features can be set in a config file found in the ```configs``` folder.

Task 1 Imputation:
- MOFA+, PoE, CGAE, Baseline have this up and running
- MoE is under construction to use other logic (see below)
- MVIB is not suitable for imputation.

Task 2 Survival Time Prediction:
- A file exists in the ```data_preprocessing``` folder that creates splits of the data to use in this task.
- Implementation will be based on the [Momix](https://github.com/ComputationalSystemsBiology/momix-notebook/blob/master/scripts/Comparison%20in%20cancer%20.ipynb) implementation, using the R ```survival``` package.
- Currently under construction
- There are also files left from cancer stage classification in the CGAE and MVAE folders.


## Implementation details

#### Multi-Omics Factor Analysis V2 (MOFA+)
The implementation of MOFA+ is provided by [MOFA+](https://biofam.github.io/MOFA2/index.html).
The code allows for the model to be trained in pure Python. This model is saved to a .hdf5 file in the appropriate directory.
Then the model's W and Z matrices (see documentation) have to be fetched using R. The Python library ```r2py``` takes care of that, but in the case of issues
there is a MOFA_downstream.R file attached.
Some notes:
- The model is trained on the training set + validation set
- Using the Moore-Penrose inverse (pseudoinverse) of the W matrix, we can multiply this with new data (Y) from a test set to get a corresponding Z for that test set.
- We can do this for both omics and then impute back from their respective Z's to the other omic's Y matrix.

#### Product-of-Experts
The MVAE code was originally adapted from the [Product-of-Experts MVAE](https://github.com/mhw32/multimodal-vae-public) as developed by Wu and Goodman.
- Their implementation has remained mostly intact. Their Product-of-Experts function in ```model.py``` and their test/training methods for example.
- The VAE architecture was changed to a more standard Vanilla-VAE architecture, based on the [Pytorch-VAE](https://github.com/AntixK/PyTorch-VAE).
- The loss function in ```train.py``` was rewritten to also work more like the Pytorch-VAE. Their loss function uses Binary cross entropy.
- Currently, this library was extended to also use a Mixture-of-Experts approach. Using all the same code but the actual combining of Gaussians. BEWARE: this code was written by what I thought was correct, but is not fully backed by a specific paper.
- To use Mixture-of-Experts, see next section.

#### Mixture-of-Experts (UNDER CONSTRUCTION)
Instead of writing an in-house implementation with chance of scrutiny, this approach will be adapted from [MMVAE](https://github.com/iffsid/mmvae). Currently, there is some work done on reusing their logic in the MVAE ```model.py``` file.
It is not yet in finalized state.

#### MVIB
Based on the following [paper](https://arxiv.org/abs/2002.07017).

#### CGAE
Some to-do's are listed in the [Drive doc](https://docs.google.com/document/d/1R6pXmTQIgCXdm_zTyRQ4GYI5oybIU4Vas66auJLRIJk/edit?usp=sharing), concerning implementation of the MultiOmicsVAE in the ```nets.py``` file.
The CGAE model is inspired by this [paper](https://www.sciencedirect.com/science/article/pii/S1046202320300232).

## File Structure
```
.
├── LICENSE
├── README.md
├── .gitignore
├── configs
│       └── geme.yaml
│       └── gegcn.yaml
│       └── gcnme.yaml
│       └── brca2_gegcn.yaml
│       └── etc.
├── data/
├── environment.yml
├── results/
├── 
├── run.py
└── src
    ├── baseline/
    ├── CGAE/
    ├── data_preprocessing/
    ├── MOFA2/
    ├── MVAE/
    ├── MVIB/
    ├── util/
    ├── nets.py
    └── survival.py
```
## Authors
    - Stavros Makrodimitris         S.Makrodimitris@tudelft.nl
    - Tamim Abdelaal                T.R.M.Abdelaal-1@tudelft.nl
    - Bram Pronk                    I.B.Pronk@student.tudelft.nl
    - Marcel Reinders               M.J.T.reinders@tudelft.nl

## Citations
- Argelaguet, R. and Velten, B. and Arnol, D. and Dietrich, S. and Zenz, T. and Marioni, J. C. and Buettner, F. and Huber, W. and Stegle, O., Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets, Mol Syst Biol, 14.6, 2018.
- Argelaguet, R. and Arnol, D. and Bredikhin, D. and Deloro, Y. and Velten, B. and Marioni, J.C. and Stegle, O.}, MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data, Genome Biology, 21.1, pp. 111, 2020.
- Section will be expanded on in the future.
