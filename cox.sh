#! /bin/sh
#SBATCH --partition=general --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000
#SBATCH --time=23:59:59
#SBATCH --job-name=survival
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL


python src/util/survival.py 'PFI' > cox_pfi.txt;
# python src/util/survival.py 'OS' > cox_os.txt;
