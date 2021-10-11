library('survival')

## Perform survival annotation-based comparison 
## INPUTS:
# factorizations = already computed factirizations
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

# Load the function running the factorization, plus a support function
source("/Users/bram/Desktop/momix-notebook-master/scripts/runfactorization.R")
source("/Users/bram/Desktop/momix-notebook-master/scripts/log2matrix.R")

# List downloaded cancer data.
# Folder structure should be organized as discussed above.
# Exclude first result as it's the parent folder
cancers <- list.dirs(path = "../data/cancer", full.names = TRUE, recursive = TRUE)[-1]
cancer_names <- list.dirs(path = "../data/cancer", full.names = FALSE, recursive = TRUE)[-1]

# Annotation databases used for biological enrichment
path.database <- "../data/bio_annotations/c2.cp.reactome.v6.2.symbols.gmt" #REACTOME
#path.database <- "../data/bio_annotations/h.all.v6.2.symbols.gmt" #Hallmarks
#path.database <- "../data/bio_annotations/c5.all.v6.2.symbols.gmt" #GO

# Label to identify current run
tag <- format(Sys.time(), "%Y%m%d%H%M%S")
# Folder for comparison results
results_folder <- paste0("../results", tag, "/")
# Create output folder
dir.create(results_folder, showWarnings = FALSE)

# Number of factors used in the paper
num.factors <- 10

# Initialize result containers
clinical_analysis <- data.frame(
  matrix(data = NA, ncol=5, nrow=0, 
         dimnames = list(c(), c("methods", "cancer", "selectivity", "nonZeroFacs", "total_pathways"))
  ),
  stringsAsFactors = FALSE)

biological_analysis <- data.frame(
  matrix(data = NA, ncol=5, nrow=0, 
         dimnames = list(c(), c("methods", "cancer", "selectivity", "nonZeroFacs", "total_pathways"))
  ),
  stringsAsFactors = FALSE)
cancer.list <- list()

# Clinical categories to be used for clinical tests
col <- c("age_at_initial_pathologic_diagnosis",
         "gender",
         "days_to_new_tumor_event_after_initial_treatment",
         "history_of_neoadjuvant_treatment")

# For each cancer dataset
for(i in cancers){
  
  print(paste0("Now analysing ", i))
  
  # Name of current cancer
  current_cancer <- basename(i)
  
  # If the expression and miRNA data are not log2-transformed as for those provided by XX et al.
  log2matrix(i,"exp")
  log2matrix(i,"mirna")
  
  # Perform factorisation
  print("Running factorisation...")
  out <- runfactorization(i, c("log_exp","methy","log_mirna"), num.factors, sep=" ", filtering="sd")
  
  # Survival analysis
  print("Running survival analysis...")
  survival <- read.table(paste0(i, "/survival"), sep="\t", header=TRUE, stringsAsFactors=FALSE)
  out_survival <- survival_comparison(out$factorizations, out$method, survival, 
                                      results_folder, current_cancer)
  
  # Clinical analysis
  print("Running clinical analysis...")
  clinical <- read.table(paste0("../data/clinical/", current_cancer), sep="\t", header=TRUE)
  out_clinical <- clinical_comparison(out$factorizations, clinical, col)   
  clinical_analysis <- rbind(clinical_analysis,
                             data.frame(methods=out$method, cancer=current_cancer, out_clinical))    
  
  # Biological analysis
  print("Running biological analysis...")
  out_bio <- biological_comparison(out$factorizations, path.database, pval.thr=0.05)
  biological_analysis <- rbind(biological_analysis,
                               data.frame(methods=out$method, cancer=current_cancer, out_bio))    
}
rownames(clinical_analysis) <- c()
rownames(biological_analysis) <- c()
