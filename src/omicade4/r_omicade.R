library(omicade4)
data(NCI60_4arrays)
GE <- read.csv("/Users/bram/jointomicscomp/data/GE.csv", nrows=50)

ME <- read.csv("/Users/bram/jointomicscomp/data/ME.csv", nrows=50)

data <- list(GE, ME)

mcoin <- mcia(data, cia.nf=5)

print(mcoin)

# General clustering
# layout(matrix(1:4, 1, 4))
# par(mar=c(2, 1, 0.1, 6))

# for (df in NCI60_4arrays) { 
#   d <- dist(t(df))
#   hcl <- hclust(d)
#   dend <- as.dendrogram(hcl)
#   plot(dend, horiz=TRUE) 
# }

# mcoin <- mcia(NCI60_4arrays, cia.nf=10)
# cancer_type <- colnames(NCI60_4arrays$agilent)
# cancer_type <- sapply(strsplit(cancer_type, split="\\."), function(x) x[1])
# plot(mcoin, axes=1:2, phenovec=cancer_type, sample.lab=FALSE, df.color=1:4)



