#! /bin/sh
#SBATCH --partition=general --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000
#SBATCH --time=22:00:00
#SBATCH --job-name=mofa
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL

# change this to your own path
DIR='/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R';

cp src/MOFA2/*.R $DIR'/';

MODALITY1='GE';
MODALITY2=$1;
echo $MODALITY1"_"$MODALITY2;

LIKELIHOOD1='normal';
LIKELIHOOD2=$2;

CATEGORIES1=-1;
CATEGORIES2=$3;


echo 'running mofa with 32 factors...'
#singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa_tcga.R 32 $MODALITY1 $MODALITY2;
echo 'running mofa with 64 factors...'
singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa_tcga.R 64 $MODALITY1 $MODALITY2;

echo 'best model and saving...'
singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofaSelectAndSave.R $MODALITY1 $MODALITY2;

cp $DIR'/mofa_tcga_factors_'$MODALITY1$MODALITY2'_training.tsv' ./src/MOFA2/;
cp $DIR'/mofa_tcga_weights_'$MODALITY1$MODALITY2'_'$MODALITY1'.tsv' ./src/MOFA2/;
cp $DIR'/mofa_tcga_weights_'$MODALITY1$MODALITY2'_'$MODALITY2'.tsv' ./src/MOFA2/;

# run downstream analysis task 1 (imputation)
./src/MOFA2/downstream.sh 1 $MODALITY1 $MODALITY2 $LIKELIHOOD1 $LIKELIHOOD2 $CATEGORIES1 $CATEGORIES2;


#MODALITY2='CNV';
#echo $MODALITY1"_"$MODALITY2;

#echo 'running mofa with 32 factors...'
#singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa_tcga.R 32 $MODALITY1 $MODALITY2;
#echo 'running mofa with 64 factors...'
#singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa_tcga.R 64 $MODALITY1 $MODALITY2;
#
#echo 'best model and saving...'
#singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofaSelectAndSave.R $MODALITY1 $MODALITY2;
#
#cp $DIR'/mofa_tcga_factors_'$MODALITY1$MODALITY2'_training.tsv' ./src/MOFA2/;
#cp $DIR'/mofa_tcga_weights_'$MODALITY1$MODALITY2'_'$MODALITY1'.tsv' ./src/MOFA2/;
#cp $DIR'/mofa_tcga_weights_'$MODALITY1$MODALITY2'_'$MODALITY2'.tsv' ./src/MOFA2/;
#
# run downstream analysis task 1 (imputation)
#./downstream.sh 1 $MODALITY1 $MODALITY2 'normal' 'categorical' -1 5;
