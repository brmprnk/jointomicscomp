#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=32000
#SBATCH --time=04:00:00
#SBATCH --job-name=mofatime
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL


for fr in 0.05 0.1 0.2 0.5 1.0;
do
	echo $fr;

	python timeMOFA.py $fr;

	echo '##################################################################################';
done

