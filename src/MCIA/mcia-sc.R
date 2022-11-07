library(omicade4)
setwd('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/')


nf <- commandArgs(trailingOnly=T)[1]
print(nf)
rna <- read.csv('sc_rna_valid.csv', header = TRUE, row.names = 1)
adt <- read.csv('sc_adt_valid.csv', header = TRUE, row.names = 1)

print('load ok')
omics_pos<-list()
rna <- rna - min(rna)
rna <- rna / max(rna)

adt <- adt - min(adt)
adt <- adt / max(adt)

omics_pos[[1]]<- t(rna)
omics_pos[[2]]<- t(adt)

start <- Sys.time()
factorizations_mcia <- mcia(omics_pos, cia.nf = as.integer(nf))
end <- Sys.time()

print(end-start)

print('saving..')
factors_mcia <- as.matrix(factorizations_mcia$mcoa$SynVar)

savedirectory <- '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/src/MCIA/'
write.csv(factors_mcia, paste0(savedirectory, 'rnaadt_train_factors_', nf, '.csv'), quote=F, row.names=T)

projection <- as.matrix(factorizations_mcia$mcoa$axis)

write.csv(projection, paste0(savedirectory, 'rnaadt_train_projection_', nf, '.csv'), quote=F, row.names=T)

