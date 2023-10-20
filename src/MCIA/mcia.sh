#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000
#SBATCH --time=04:00:00
#SBATCH --job-name=mcia
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL

#Rscript mcia.R $1;
#Rscript mcia-sc.R $1;
#Rscript mcia-cnv.R $1;
#Rscript mcia-atac.R $1;
/tudelft.net/staff-umbrella/liquidbiopsy/momix/bin/Rscript mcia-atac.R $1;
