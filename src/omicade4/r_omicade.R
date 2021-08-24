library(omicade4)

GE <- read.csv("/Users/bram/jointomicscomp/data/GE.csv", nrows=50)

ME <- read.csv("/Users/bram/jointomicscomp/data/ME.csv", nrows=50)

data <- list(GE, ME)

mcoin <- mcia(data, cia.nf=5)

print(mcoin)
