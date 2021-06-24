# CSE3000 Research Project - Bram Pronk - Group 21
### Assessing How Variational Auto-Encoders Can Combine Information From Multiple Data Sources in Cancer Cells

[![Python Version](https://img.shields.io/static/v1.svg?label=minimal_python_version&message=3.8.8&color=blue)](https://www.python.org/downloads)

## Research Abstract
Personalized treatment methods for a complex disease such as cancer benefit from using multiple data modalities from a patient's tumor cells. Multiple modalities allow for analysis of dependencies between complex biological processes and downstream tasks, such as drug response and/or expected survival rate. To this end, it is important to gain an understanding of the relationships between modalities in tumor cells. Multimodal Variational Auto-Encoders (MVAEs) are a combination of generative models trained on different sets of data modalities. In this research, the ability of MVAEs to capture common information between different data views from the same tumor cells is assessed. MVAE models discussed here are a Mixture-of-Experts (MoE) and a Product-of-Experts (PoE) approach to combining the generative model posterior distributions into a single common latent space. The performance assessment is done by: i) comparing the loss of information when reconstructing the training data to MOFA+, a linear method for combining multimodal data, and  ii) measuring if one modality of a tumor cell can generate another modality, based on characteristics of the latent space learned by the MVAE. Biological data modalities considered are RNA-seq, gene-level copy number and DNA methylation (DNAme), gathered by The Cancer Genome Atlas. It is found that PoE reconstructs data from all modalities with a higher accuracy compared to MoE and MOFA+. The mean squared error of PoE's average reconstruction loss is about a quarter of MOFA+'s, and less than a seventh of the MoE's average reconstruction loss. In terms of predicting modalities from other modalities, the PoE again outperforms MoE on all cross-modal predictions. Additionally, it can be concluded that both models have higher losses in their prediction of DNAme from other modalities, indicating a lesser correlation between this modality and the others.

## Getting Started
<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->
This project will compare linear method Multi-Omics Factor Analysis V2 (MOFA+) with two proposed models for VAE's that span multiple data modalities (MVAE), namely Mixture-of-Experts (MoE) and Product-of-Experts (PoE). 
To that end, this repository will have three branches for each model; ```mofa```, ```moe``` and ```poe```. 
Since each model will be a fork from an existing repository that is adjusted to fit TCGA's data, environment setup will differ. To that end, instructions for installation and running will be provided in the README.md of each branch.
The ```dev``` branch will contain files relevant to all models, such as data preprocessing and result plots.

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
