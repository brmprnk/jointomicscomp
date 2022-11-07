library(MOFA2)

modality1 <- commandArgs(trailingOnly=T)[1]
modality2 <- commandArgs(trailingOnly=T)[2]


model32 <- load_model(paste0('/mnt/tcga-model_', modality1, '_', modality2, '_32.hdf5'))
model64 <- load_model(paste0('/mnt/tcga-model_', modality1, '_', modality2, '_64.hdf5'))

sel <- select_model(c(model32, model64))
fac <- get_factors(sel)

write.table(fac$group1, paste0("/mnt/mofa_tcga_factors_", modality1, modality2, "_training.tsv"), quote=F, sep="\t")


weights <- get_weights(sel)

write.table(weights$rna, paste0("/mnt/mofa_tcga_weights_", modality1, modality2, "_", modality1, ".tsv"), quote=F, sep="\t")
write.table(weights$adt, paste0("/mnt/mofa_tcga_weights_", modality1, modality2, "_", modality2, ".tsv"), quote=F, sep="\t")
