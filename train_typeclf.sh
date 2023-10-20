#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --time=00:45:00
#SBATCH --job-name=typeclf
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL


ml use /opt/insy/modulefiles;
ml load cuda/11.0;
ml load cudnn/11.0-8.0.3.33;


python src/util/trainTypeClassifier.py 'configs/type-clf/RNA2.yaml';
python src/util/trainTypeClassifier.py 'configs/type-clf/ATAC.yaml';
#for CONFIG_FILE in $(ls configs/type-clf/);
#do
# echo 'Training from '$CONFIG_FILE;
# python src/util/trainTypeClassifier.py 'configs/type-clf/'$CONFIG_FILE;
#done
