library(MOFA2)
setwd('/mnt')

nf <- as.integer(commandArgs(trailingOnly=T)[1])
modality1 <- commandArgs(trailingOnly=T)[2]
modality2 <- commandArgs(trailingOnly=T)[3]



print(nf)
rna <- read.csv(paste0('/mnt/', modality1, '_train.csv'), header = TRUE, row.names = 1)
adt <- read.csv(paste0('/mnt/', modality2, '_train.csv'), header = TRUE, row.names = 1)

print('load ok')

omics_pos<-list()
omics_pos$rna <- t(rna)
omics_pos$adt <- t(adt)


model <- create_mofa(omics_pos)

model_opts <- get_default_model_options(model)
model_opts$num_factors <- nf

train_opts <- get_default_training_options(model)
#train_opts$convergence_mode <- "medium"
train_opts$seed <- 1

model <- prepare_mofa(object = model, data_options = get_default_data_options(model), model_options = model_opts, training_options = train_opts)

outfile = file.path(getwd(), paste0("tcga-model_", modality1, "_", modality2, "_", nf, ".hdf5"))
print('starting to run mofa')
trainedModel <- run_mofa(model, outfile)
print('done')
