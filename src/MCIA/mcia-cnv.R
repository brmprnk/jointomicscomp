library(omicade4)
setwd('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R/')


nf <- commandArgs(trailingOnly=T)[1]
print(nf)
ge <- read.csv('GE_train.csv', header = TRUE, row.names = 1)
me <- read.csv('CNV_train.csv', header = TRUE, row.names = 1)

print('load ok')
omics_pos<-list()
ge <- ge - min(ge)
ge <- ge / max(ge)


omics_pos[[1]]<- t(ge)
omics_pos[[2]]<- t(me)

start <- Sys.time()
factorizations_mcia<-mcia(omics_pos, cia.nf = as.integer(nf))
end <- Sys.time()

print(end-start)

print('saving..')
factors_mcia <- as.matrix(factorizations_mcia$mcoa$SynVar)

savedirectory <- '/tudelft.net/staff-umbrella/liquidbiopsy/neural-nets/jointomicscomp/src/MCIA/'
write.csv(factors_mcia, paste0(savedirectory, 'gecnv_train_factors_', nf, '.csv'), quote=F, row.names=T)

projection <- as.matrix(factorizations_mcia$mcoa$axis)

write.csv(projection, paste0(savedirectory, 'gecnv_train_projection_', nf, '.csv'), quote=F, row.names=T)

