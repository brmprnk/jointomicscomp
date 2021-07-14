# CSE3000 Research Project - Bram Pronk Bachelor Thesis

[![Python Version](https://img.shields.io/static/v1.svg?label=minimal_python_version&message=3.8.8&color=blue)](https://www.python.org/downloads)

## Research Paper
Github Repository that accompanies the research for my bachelor thesis: 
http://resolver.tudelft.nl/uuid:4c23e8d7-0fe6-4dd8-9c38-e62a39c07a99

## Branch Structure & Getting Started
<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->
#### Follow-up research will be done in the branch ```followup```.

This project compares linear method Multi-Omics Factor Analysis V2 (MOFA+) with two proposed models for VAE's that span multiple data modalities (MVAE), namely Mixture-of-Experts (MoE) and Product-of-Experts (PoE). 
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
