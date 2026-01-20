# Largescale Urban Health Demo

# Overview

We present an open science workflow combining CDC Places Health data with explainable graph deep learning to understand the nexus between urban contextual and social determinants of health factors for general, physical, and mental health outcomes across the most populous urban areas within the contiguous United States. The repository includes scripts to process data inputs and reproduce the main findings and figures. 

# System Requirements

## Hardware Requirements

To run the demo workflow, users require only a standard computer with enough RAM to support the operations defined by a user. Having >4GB RAM would be preferable. 

The demo was implemented on a MacBook Pro, M1 Chip with 16GB RAM and 512GB storage.

## Software Requirements

### OS Requirements

The package development version is tested on MacOS operating systems. The implementation would work on the following systems:

Linux: Ubuntu 22.04  
Mac OSX:  
Windows:  

Before running the demo, users should have Python version 3.8.0 or higher.

# Installation Guide & Dependencies

1. Clone the code repository either with CLI or by accessing the code base
2. Navigate to the local folder location
3. Open up a terminal/command prompt and install the conda environment

```
$ cd ./<location>
$ chmod +x ./setup.sh
$ ./setup.sh
```

4. Installation completed

(Optional) For JupyterLab / JupyterNotebook users, you can additionally add a notebook kernel via:

```
$ conda activate carbon_demo
```

# Repository Structure
- `data` (Placeholder folder for datasets [^1])
- `figures` (Placeholder folder storing figures)

[^1]: Data repository storing preprocessed data objects. Due to github file size storage limits, the data files are hosted on Figshare: https://figshare.com/articles/dataset/Accompanying_Data_Package_for_Largescale_Urban_Health/31094953 

# Quickstart

## To generate city graphs
Please check out the `notebooks` folder in this directory. 

## To generate data figures
1) Download files from data repository and save them to `data` folder
2) run `python main.py` in command line

<br>

# Citation

If you use Urbanity in your work, please cite:
<br></br>
Yap, W., Stouffs, R. & Biljecki, F. Urbanity: automated modelling and analysis of multidimensional networks in cities. npj Urban Sustain 3, 45 (2023). https://doi.org/10.1038/s42949-023-00125-w

Yap, W., Biljecki, F. A Global Feature-Rich Network Dataset of Cities and Dashboard for Comprehensive Urban Analyses. Sci Data 10, 667 (2023). https://doi.org/10.1038/s41597-023-02578-1

Yap, W., Wu, A. N., Miller, C., & Biljecki, F. (2025). Revealing building operating carbon dynamics for multiple cities. Nature Sustainability, 1-12. https://doi.org/10.1038/s41893-025-01615-8

Yap, W., Duarte, F., Zheng, Y., Jang., K. M., Luo, P., Ratti, C., & Biljecki, F. (Under review). Disentangling Human-Environment Pathways to General, Physical, and Mental Health.