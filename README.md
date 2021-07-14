# CSE3000 Research Project - Bram Pronk Bachelor Thesis

[![Python Version](https://img.shields.io/static/v1.svg?label=minimal_python_version&message=3.9&color=blue)](https://www.python.org/downloads)
[![Conda Install](https://anaconda.org/conda-forge/terraform-provider-github/badges/installer/conda.svg)](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
## Research Paper
Github Repository that accompanies the research for my bachelor thesis: 
http://resolver.tudelft.nl/uuid:4c23e8d7-0fe6-4dd8-9c38-e62a39c07a99

#### Follow-up research is be done in this branch ```followup```.
Original suggestions for follow up research included:
- Implementing Mixture-of-Experts and Product-of-Experts in the same codebase
- Switching from Vanilla-VAE models to Beta-VAE to possibly improve learning, and investigating the consequences in doing so

## Installation
Recommended installation uses a new Anaconda environment. To ease the process, this project includes an environment file.
This can be plugged into Anaconda [following this short tutorial](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## Branch Structure
<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->


This branch builds upon the code found in the ```poe``` branch.
This code was originally adapted from the [Product-of-Experts MVAE](https://github.com/mhw32/multimodal-vae-public) as developed by Wu and Goodman.

It has been expanded to include multiple optional arguments, that can be found by running the main file ``main.py`` with the `-h` option.



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
