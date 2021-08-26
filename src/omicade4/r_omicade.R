library(omicade4)
library("ggplot2")

GE <- read.csv("/Users/bram/jointomicscomp/data/GE.csv", nrows=50)

ME <- read.csv("/Users/bram/jointomicscomp/data/ME.csv", nrows=50)

data <- list(GE, ME)
num_factors <- 5

factorizations_mcia <- mcia(data, cia.nf=num_factors)

# From Momix
# https://github.com/ComputationalSystemsBiology/momix-notebook/blob/master/scripts/runfactorization.R

factors_mcia<-as.matrix(factorizations_mcia$mcoa$SynVar)

metagenes_mcia<-list()
for(j in 1:length(data)){
  metagenes_mcia[[j]]<-as.matrix(factorizations_mcia$mcoa$axis[1:dim(data[[j]])[1],])
  rownames(metagenes_mcia[[j]])<-rownames(data[[j]])
  colnames(metagenes_mcia[[j]])<-1:num_factors
}

