#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000
#SBATCH --time=04:00:00
#SBATCH --job-name=mofa
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL

# change this to your own path
DIR='/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R';

cp src/MOFA2/*.R $DIR'/';

MODALITY1='RNA';
MODALITY2='ATAC';
echo $MODALITY1"_"$MODALITY2;

LIKELIHOOD1='nb';
LIKELIHOOD2='bernoulli';
# LIKELIHOOD1='normal';
# LIKELIHOOD2='normal';

CATEGORIES1=-1;
CATEGORIES2=-1;


echo 'running mofa with 32 factors...'
singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa_atac.R 32;
echo 'running mofa with 64 factors...'
singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa_atac.R 64;
exit;
echo 'best model and saving...'
singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofaSelectAndSave.R $MODALITY1 $MODALITY2;

cp $DIR'/mofa_atac_factors_'$MODALITY1$MODALITY2'_training.tsv' ./src/MOFA2/;
cp $DIR'/mofa_atac_weights_'$MODALITY1$MODALITY2'_'$MODALITY1'.tsv' ./src/MOFA2/;
cp $DIR'/mofa_atac_weights_'$MODALITY1$MODALITY2'_'$MODALITY2'.tsv' ./src/MOFA2/;

# run downstream analysis task 1 (imputation)
#./src/MOFA2/downstream.sh 2 $MODALITY1 $MODALITY2 $LIKELIHOOD1 $LIKELIHOOD2 $CATEGORIES1 $CATEGORIES2;
