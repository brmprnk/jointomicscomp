#! /bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mofa
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL

# change this to your own path
DIR='/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/ge-me-cn-2022-04-16/R';

cp src/MOFA2/*.R $DIR'/';

singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa.R 32;
singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofa.R 64;

singularity run --bind $DIR:/mnt ~/mofa2_latest.sif Rscript /mnt/mofaSelectAndSave.R;

cp $DIR'/mofa_cite_factors_training.tsv' ./src/MOFA2/;
cp $DIR'/mofa_cite_weights_rna.tsv' ./src/MOFA2/;
cp $DIR'/mofa_cite_weights_adt.tsv' ./src/MOFA2/;

# run downstream analysis task 2 (imputation & classification)
./downstream.sh 2;
