#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000
#SBATCH --time=01:00:00
#SBATCH --job-name=survival
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL


python src/util/survival.py $1
