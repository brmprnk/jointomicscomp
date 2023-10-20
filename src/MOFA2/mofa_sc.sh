#! /bin/sh
#SBATCH --partition=general --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=08:00:00
#SBATCH --job-name=mofa
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL

#python src/MOFA2/mofa_cite_gpu.py 16;
python src/MOFA2/mofa_cite_gpu.py $1;
#python src/MOFA2/mofa_cite_gpu.py 64;

# cp $DIR'/mofa_cite_factors_training.tsv' ./src/MOFA2/;
# cp $DIR'/mofa_cite_weights_rna.tsv' ./src/MOFA2/;
# cp $DIR'/mofa_cite_weights_adt.tsv' ./src/MOFA2/;
#
# # run downstream analysis task 2 (imputation & classification)
# ./downstream.sh 2;
