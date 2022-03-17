#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000
#SBATCH --time=04:00:00
#SBATCH --job-name=evalJoint
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL


python src/util/disentanglement.py 'PFI' > cox_pfi.txt;
python src/util/disentanglement.py 'OS' > cox_os.txt;
