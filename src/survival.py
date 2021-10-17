"""
Main file for running Survival Prediction.
Based on the survival prediction code by Momix
https://github.com/ComputationalSystemsBiology/momix-notebook/blob/master/scripts/Comparison%20in%20cancer%20.ipynb

Momix code is written in R, and therefore this file uses rpy2.
RUN FACTORIZATIONS BEFOREHAND.
"""
import os
# R needs to be installed, and this path needs to be set to the R_Home folder, found by running R.home() in R console.
os.environ['R_HOME'] = "/usr/lib/R"
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def run(args: dict) -> None:
    """
    Setup before running Survival prediction.

    @param args: Dictionary containing input parameters
    @return: None
    """
    save_dir = os.path.join(args['save_dir'], 'survival')
    os.makedirs(save_dir)

    # Take factorizations:
    z = np.load("/home/bram/jointomicscomp/results/MVAE Task 1 FINAL GEGCN/PoE/task1_z.npy")

    survival(z, save_dir=save_dir)


def survival(z, save_dir: str) -> None:
    """
    Perform R based code here in Python.
    Documentation: https://rpy2.github.io/doc/v2.9.x/html/introduction.html

    @param save_dir:   path to directory where factors and weights should be saved

    @return: None
    """
    # import R's "base" package
    rpackages.importr('base')

    # import R's "utils" package
    utils = rpackages.importr('utils')

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

    # MOFA required R package names
    package_names = ('ggplot2', 'MOFA2')

    # R vector of strings
    from rpy2.robjects.vectors import StrVector

    # Selectively install what needs to be installed
    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    robjects.r('''
            library('survival')
            
            ## Perform survival annotation-based comparison 
            ## INPUTS:
            # factorizations = already computed factorizations
            # method = methods used for factorization
            # survival = survival data associated to the cancer
            # out.folder = folder where results will be written
            # cancer = name of currently analysed cancer
            ## OUPUTS: a list containing output values
            survival_comparison <- function(factorizations, method, survival, out.folder, cancer){
                
                # Initialize result containers
                factors_cancer <- numeric(0)
                surv_final <- numeric(0)
                
                # Adjust sample names for breast survival dataset
                if(cancer=="breast"){survival[,1] <- paste0(survival[,1],"-01")}
                
                # For each computed factorisation
                for(i in 1:length(factorizations)){
            
                    # Extract sample factors 
                    factors <- factorizations[[i]][[1]]
            
                    # Patient names in factorisation results
                    patient.names <- rownames(factors)
                    # Patient names in original data
                    patient.names.in.file <- as.character(survival[, 1])
                    patient.names.in.file <- toupper(gsub('-', '\\.', patient.names.in.file))
                    # Remove non-matching patient names
                    is_in_file <- patient.names %in% patient.names.in.file
                    if(length(patient.names)!=sum(is_in_file)) {
                        factors <- factors[is_in_file, ]
                        patient.names <- patient.names[is_in_file]
                        rownames(factors)<-patient.names
                    }
                    # Match indices of patient names
                    indices <- match(patient.names, patient.names.in.file)
                    # Use indices to extract coresponding survival information
                    ordered.survival.data <- survival[indices,]
                    # Clean data (assign 0 to NAs)
                    ordered.survival.data$Survival[is.na(ordered.survival.data$Survival)] <- 0
                    ordered.survival.data$Death[is.na(ordered.survival.data$Death)] <- 0
            
                    # Calculate coxph
                    coxph_obj <- coxph(Surv(ordered.survival.data$Survival, ordered.survival.data$Death) ~ factors)
                    # P-values (corrected by the number of methods)
                    pvalues <- length(factorizations)*as.matrix(coef(summary(coxph_obj))[,5])
            
                    # How many significant? 
                    factors_cancer <- c(factors_cancer, sum(pvalues<0.05))
                    # Store p-values
                    surv_final <- cbind(surv_final, pvalues) 
                }
                # Keep -log10 of p-values
                surv_final<-(-log10(surv_final))
                # Plot survival pvalues for each cancer type separately
                png(file=paste0(out.folder, "survival_", cancer, ".png"), width = 15, height = 15, units = 'in', res = 200)
                matplot(1:length(method), t(surv_final), 
                        col="black", pch=18, xlab="Method", ylab="Pvalues survival", xaxt="none", cex=1.5)
                abline(h = (-log10(0.05)), v=0, col="black", lty=3, lwd=3)
                axis(1, at=1:length(method), labels=colnames(surv_final)) 
                dev.off()
            
            #    print(factors_cancer)
                return(factors_cancer)
            }

            ''')

    nr, nc = z.shape
    print(nr, nc)
    Br = rpy2.robjects.r.matrix(z, nrow=nr, ncol=nc)

    rpy2.robjects.r.assign("z", Br)

    print(Br)

    # run R code
    survival_comparison = robjects.globalenv['survival_comparison']
    # survival_comparison(save_dir, model_file)