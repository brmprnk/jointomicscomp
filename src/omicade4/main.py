"""
Main file for running Omicade 4
https://bioconductor.org/packages/release/bioc/html/omicade4.html

This library is implemented in R.
To get in the same workflow as the other models, Python will run this script.
MAKE SURE IS INSTALLED ON YOUR MACHINE
The R_Home folder can be found by running R.home() in R console, and needs to be added on the top import.
"""
import os
# R needs to be installed, and this path needs to be set to the R_Home folder, found by running R.home() in R console.
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from src.util import logger


def run_omicade(args: dict) -> None:
    """
    Perform R based code here in Python.
    Documentation: https://rpy2.github.io/doc/v2.9.x/html/introduction.html

    @return: None
    """
    logger.info("Running Omicade4")
    logger.success("Now showing output from the R console : ")

    # Setup output paths
    save_dir = os.path.join(args['save_dir'], 'omicade4')
    os.makedirs(save_dir)

    # import R's "base" package
    rpackages.importr('base')

    # import R's "utils" package
    utils = rpackages.importr('utils')

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

    # Omicade4 required R package names
    package_names = ('ggplot2', 'BiocManager', 'omicade4')

    # R vector of strings
    from rpy2.robjects.vectors import StrVector

    # Selectively install what needs to be installed
    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    robjects.r('''
            library(omicade4)
            data(NCI60_4arrays)
            
            # Check for right ordering of samples
            if (all(apply((x <- sapply(NCI60_4arrays, colnames))[,-1], 2, function(y)
                + identical(y, x[,1])))) {
                print("The samples in both datasets are ordered correctly")   
            } else {
                stop("Wrong ordering in datasets")
            }
    
            # create a function `run_mcia` that performs the MCIA
            run_mcia <- function(result_dir, data, cia_nf=10) {
                setwd(result_dir)

                # General clustering
                png("hierarchical_clustering.png")
                layout(matrix(1:4, 1, 4))
                par(mar=c(2, 1, 0.1, 6))

                for (df in NCI60_4arrays) { 
                    d <- dist(t(df))
                    hcl <- hclust(d)
                    dend <- as.dendrogram(hcl)
                    plot(dend, horiz=TRUE)
                }
                dev.off()
                
                png("mcia_plot.png")
                mcoin <- mcia(NCI60_4arrays, cia.nf=10)
                cancer_type <- colnames(NCI60_4arrays$agilent)
                cancer_type <- sapply(strsplit(cancer_type, split="\\\."), function(x) x[1])
                plot(mcoin, axes=1:2, phenovec=cancer_type, sample.lab=FALSE, df.color=1:4)
                dev.off()
            }
            ''')

    r_run_mcia = robjects.globalenv['run_mcia']

    # Save model factors and weights (Z and W)
    r_run_mcia(save_dir, [], 10)
