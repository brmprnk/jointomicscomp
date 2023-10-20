library(MOFA2)
setwd('/mnt')

nf <- as.integer(commandArgs(trailingOnly=T)[1])
#nf <- 64
print(nf)

rna <- read.csv('/mnt/atac_rna_train.csv', header = TRUE, row.names = 1)
atac <- read.csv('/mnt/atac_atac_train.csv', header = TRUE, row.names = 1)

rna <- log(rna + 1)

print('load ok')

omics_pos<-list()
omics_pos$rna <- t(rna)
omics_pos$atac <- t(atac)


model <- create_mofa(omics_pos)

model_opts <- get_default_model_options(model)
model_opts$num_factors <- nf

mydistr<-c("gaussian","bernoulli")
names(mydistr)<-c("rna","atac")
model_opts$likelihoods<-mydistr

train_opts <- get_default_training_options(model)
train_opts$convergence_mode <- "slow"
train_opts$seed <- 1


model <- prepare_mofa(object = model, data_options = get_default_data_options(model), model_options = model_opts, training_options = train_opts)

outfile = file.path(getwd(), paste0("atac-model_", nf, ".hdf5"))
print('starting to run mofa')
trainedModel <- run_mofa(model, outfile)
print('done')
