# CSE3000 Research Project - Bram Pronk - Group 21 | MOFA
### Assessing How Variational Auto-Encoders Can Combine Information From Multiple Data Sources in Cancer Cells

[![Python Version](https://img.shields.io/static/v1.svg?label=minimal_python_version&message=3.8.8&color=blue)](https://www.python.org/downloads)

## Description of the MOFA in this research
See Research Paper Section 3.1.
Principal component analysis (PCA) is one of the most common linear dimensionality reduction algorithms. Though it lies at the basis of this method, it needs alterations order to suit data from multiple modalities. Multi-Omics Factor Analyis V2 (MOFA+) can in that sense be described as "a versatile and statistically rigorous generalization of principal component analysis to multi-omics data". MOFA+ infers a low-dimensional latent space from a high-dimensional set of data. Each modality is presented to the algorithm as a separate view (Y_1 ... Y_m). Each sample is then decomposed into ten factors and this low-dimensional representation is presented as the Z matrix. Ten factors were chosen since previous research on RNA-seq and DNA methylation showed 10 factors explained all the variance. Then "for each factor, the weights (W) link the high-dimensional space with the low-dimensional manifold and provide a measure of feature importance". Further details are withheld here but are explained thoroughly in the original paper.
## Getting Started
<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->
This part of project will perform a linear Multi-Omics Factor Analysis (MOFA) on [TCGA's RNA-seq data set](https://xenabrowser.net/datapages/?dataset=EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena&host=https%3A%2F%2Fpancanatlas.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443). The implementation of MOFA is provided by [MOFA+](https://biofam.github.io/MOFA2/index.html). MOFA+ requires a specific view of the dataset. To that end, this branch contains file ```create_mofa_data.py``` to convert .xena file or .csv file of the TCGA dataset to the required MOFA+ DataFrame.
 
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