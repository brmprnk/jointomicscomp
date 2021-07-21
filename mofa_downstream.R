library(ggplot2)
library(MOFA2)

trained_model <- ("/Users/bram/rp-group-21-bpronk-mofa/trained_models/shuffle_clamped_RNA_GCN_DNA_Trained_MOFA_10Factors.hdf5")

model <- load_model(trained_model, remove_inactive_factors = F)

Z = get_expectations(model, "Z", as.data.frame = TRUE)
W = get_expectations(model, "W", as.data.frame = TRUE)
write.csv(Z, "/Users/bram/rp-group-21-bpronk-mofa/data/shuffle_Z.csv", row.names = FALSE)
write.csv(W, "/Users/bram/rp-group-21-bpronk-mofa/data/shuffle_W.csv", row.names = FALSE)

plot_factor_cor(model)

plot_data_overview(model)

plot_variance_explained(model, x="view", y="factor")

plot_variance_explained(model, plot_total = TRUE)[[2]]
plot_weights(model,
             view = "RNA",
             factor = 2,
             nfeatures = 10,     # Top number of features to highlight
)

# Prediction code using MOFA+ predict(model)
# predictions <- predict(model)
# recon_rna = predictions['RNA']
# rna_frame = data.frame(recon_rna)
# write.csv(rna_frame, "/Users/bram/rp-group-21-bpronk/data/rna_predict.csv", row.names = FALSE)
# recon_gcn = predictions['GENE_CN']
# gcn_frame = data.frame(recon_gcn)
# write.csv(gcn_frame, "/Users/bram/rp-group-21-bpronk/data/gcn_predict.csv", row.names = FALSE)
