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
python run.py -c configs/main.yaml
```
This default call to run.py will run all implemented models. For specific models, use one or combine the 
```-poe -moe -mofa -mvib -cgae``` flags.
All results and model outputs will be stored in the project's ```results``` directory.

To see additional arguments, use argparse's built in -h command:
```bash
python run.py -h
```

## Implementation details
- MOFA+: Multi-Omics Factor Analysis V2 (MOFA+).
  The implementation of MOFA+ is provided by [MOFA+](https://biofam.github.io/MOFA2/index.html).
  MOFA+ requires a specific view of the dataset. To that end, this repo contains the file ```create_mofa_data.py``` to convert .xena file or .csv files to the required MOFA+ DataFrame.
 
- Product-of-Experts: The MVAE code was originally adapted from the [Product-of-Experts MVAE](https://github.com/mhw32/multimodal-vae-public) as developed by Wu and Goodman.

## File Structure
```
.
├── LICENSE
├── README.md
├── configs
│       └── main.yaml
├── data/
├── definitions.py
├── environment.yml
├── results/
├── run.py
└── src
    ├── MVAE
    │   ├── datasets.py
    │   ├── model.py
    │   ├── pca.py
    │   ├── predict.py
    │   ├── train.py
    │   └── umapz.py
    ├── data_preprocessing
    │   ├── TCGA_data_preprocessing.py
    │   ├── TCGA_sort_and_shuffle_data_by_cancertype.ipynb
    │   └── TCGA_three_cancertypes.ipynb
    └── util
        └── MVAE_plotting.py
```
## Authors
    - Stavros Makrodimitris         S.Makrodimitris@tudelft.nl
    - Marcel Reinders               M.J.T.reinders@tudelft.nl
    - Bram Pronk                    I.B.Pronk@student.tudelft.nl

## Citations
- Argelaguet, R. and Velten, B. and Arnol, D. and Dietrich, S. and Zenz, T. and Marioni, J. C. and Buettner, F. and Huber, W. and Stegle, O., Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets, Mol Syst Biol, 14.6, 2018.
- Argelaguet, R. and Arnol, D. and Bredikhin, D. and Deloro, Y. and Velten, B. and Marioni, J.C. and Stegle, O.}, MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data, Genome Biology, 21.1, pp. 111, 2020.
