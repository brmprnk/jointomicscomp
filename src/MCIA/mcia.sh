#! /bin/sh
#SBATCH --partition=general --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=08:00:00
#SBATCH --job-name=mcia
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL

#Rscript mcia.R $1;
#Rscript mcia-sc.R $1;
Rscript mcia-cnv.R $1;
