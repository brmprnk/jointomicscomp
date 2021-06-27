# CSE3000 Research Project - Bram Pronk - Group 21 | MOFA+
### Assessing How Variational Auto-Encoders Can Combine Information From Multiple Data Sources in Cancer Cells

[![Python Version](https://img.shields.io/static/v1.svg?label=minimal_python_version&message=3.8.8&color=blue)](https://www.python.org/downloads)

## Description of MOFA+ in this research
See Research Paper Section 3.1.
Link to paper will be uploaded after submission has finalized.

## Getting Started
<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->
This part of project will perform a linear Multi-Omics Factor Analysis V2 (MOFA+) on TCGA's RNA-seq, Gene level copy number (Gistic2), and DNA Methylation 450K data. The implementation of MOFA+ is provided by [MOFA+](https://biofam.github.io/MOFA2/index.html). MOFA+ requires a specific view of the dataset. To that end, this branch contains file ```create_mofa_data.py``` to convert .xena file or .csv file of the TCGA dataset to the required MOFA+ DataFrame.

Downstream analysis is done in R. Running ```mofa_downstream.R``` will write files to the project's directory which are used for calculation of the reconstruction loss in ```recon_loss.py``` and latent space UMAP in ```umapz.py```.

Highly recommended to use a fresh Anaconda environment for this branch. A pre-set up environment file is added for one click setup : ```mofa_conda_environment.yml```.
 
## Branch Structure
```
|---trained_models/
    |---All_cancertypes.hdf5
    |---3_cancertypes.hdf5
|
|---README.md
|---mofa.py
|---create_mofa_data.py
|---mofa_downstream.R
|---recon_loss.py
|---umapz.py
```

## Authors
This is the personal repository of

    - Bram Pronk            (4613066) i.b.pronk@student.tudelft.nl

My colleagues in Research Group 21:

     - Armin Korkic         (4713052) a.korkic@student.tudelft.nl
     - Boris van Groeningen (4719875) b.vangroeningen@student.tudelft.nl
     - Ivo Kroskinski       (4684958) i.s.kroskinski@student.tudelft.nl
     - Raymond d'Anjou      (4688619) r.danjou@student.tudelft.nl

Research Group 21 is guided by:
    
    - Marcel Reinders (Responsible Professor)
    - Stavros Makrodimitris, Tamim Abdelaal, Mohammed Charrout, Mostafa elTager (Supervisors)

## Citations
Argelaguet, R. and Velten, B. and Arnol, D. and Dietrich, S. and Zenz, T. and Marioni, J. C. and Buettner, F. and Huber, W. and Stegle, O., Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets, Mol Syst Biol, 14.6, 2018.

Argelaguet, R. and Arnol, D. and Bredikhin, D. and Deloro, Y. and Velten, B. and Marioni, J.C. and Stegle, O.}, MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data, Genome Biology, 21.1, pp. 111, 2020.
