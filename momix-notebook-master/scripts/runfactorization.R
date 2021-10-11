##### runfactoization runs all the considered multi-omics factorization
### the required inputs are:
### "folder" corresponding to the path to the folder where the input files are contained, in the idea that all omics matrices are organized inside a unique folder
### "file.names" corresponding to a vector containing the names of all the omics files
### "num.factors" containing the number of factors in which we desire to decompose the matrices
### "sep=" "" corresponding to the separator used in the omics files required to properly read them
### "single.cell" indicating if the data are single cell data. In this case the filtering of the data will be more intense in respect to other data types
### "filtering"

### the input files need to be log2 transformed before running the analysis


library("RGCCA")
library("r.jive")
library("IntNMF")
library("omicade4")
#library("MSFA")
#library("GPArotation")
#library("MOFAtools")
#library("tensorBSS")
#source("tICA.R")
#library("iCluster")

runfactorization <- function(folder,file.names,num.factors,sep=" ",filtering="none"){
  factorizations<-list()
  t<-1
  method<-numeric(0)
  
  num.factors<-as.numeric(num.factors)

  
  ##creating list of omics
  omics <- list()
  for(i in 1:length(file.names)){
    omics[[i]]<-as.matrix(read.table(paste(folder,file.names[i],sep="/"),sep=sep,row.names=1,header=T))
  }
  
  ####
  #omics<-lapply(omics, function(x) t(x))
  ######
  
  ##restricting to common samples and filtering
  samples<-colnames(omics[[1]])
  for(j in 1:length(omics)){
    samples<-intersect(samples,colnames(omics[[j]]))
  }
  for(j in 1:length(omics)){
    omics[[j]]<-omics[[j]][,samples]
    if(filtering!="none"){
      x<-apply( omics[[j]],1,sd)
      x<-as.matrix(sort(x, decreasing = T))
      w<-which(x>0)
      if(filtering=="stringent"){
        selected<-rownames(x)[1:min(w[length(w)],5000)]
      }else{
        selected<-rownames(x)[1:min(w[length(w)],6000)]
      }
      m<-match(rownames(omics[[j]]),selected)
      w<-which(!is.na(m))
      omics[[j]]<-omics[[j]][w,]
    }else{
      omics[[j]]<-omics[[j]][which(apply(omics[[j]],2,sd)>0),]
    }
  }  
  
  ###MCIA
  omics_pos<-list()
  for(j in 1:length(omics)){
    if(min(omics[[j]])<0){
      omics_pos[[j]]<-omics[[j]]+abs(min(omics[[j]]))
    }else{
      omics_pos[[j]]<-omics[[j]]
    }
    omics_pos[[j]]<-omics_pos[[j]]/max(omics_pos[[j]])
  }
  factorizations_mcia<-mcia(omics_pos, cia.nf = num.factors)
  factors_mcia<-as.matrix(factorizations_mcia$mcoa$SynVar)
  metagenes_mcia<-list()
  for(j in 1:length(omics)){
    metagenes_mcia[[j]]<-as.matrix(factorizations_mcia$mcoa$axis[1:dim(omics[[j]])[1],])
    rownames(metagenes_mcia[[j]])<-rownames(omics[[j]])
    colnames(metagenes_mcia[[j]])<-1:num.factors
  }
  factorizations[[t]]<-list(factors_mcia,metagenes_mcia)
  t<-t+1
  method<-c(method,"MCIA")
  
  
  out<-list(factorizations=factorizations,method=method,icluster.clusters=as.matrix(factorizations_icluster$clusters),intNMF.clusters=as.matrix(factorizations_intnmf$clusters))
  
  return(out)
}
