#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --time=00:30:00
#SBATCH --job-name=trainJoint
#SBATCH --mail-user=bpronk
#SBATCH --mail-type=ALL

# conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/env-vae/vae
ml use /opt/insy/modulefiles;
ml load cuda/11.0;
ml load cudnn/11.0-8.0.3.33;


# DoubleConstraintAutoEncoder:				'dcae'
# LikelihoodAutoencoder:							'lae'
# VariationalAutoencoder:							'vae'
# LikelihoodVariationalAutoencoder:		'lvae'

CONFIG_FILE='configs/geme.yaml'

echo 'start';
TRAIN_LOG="train-log.txt";
python run.py --experiment='test' --config=${CONFIG_FILE} >> ${TRAIN_LOG};
