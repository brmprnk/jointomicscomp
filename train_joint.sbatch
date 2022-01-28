#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --time=00:30:00
#SBATCH --job-name=trainJoint
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=FAIL

## $1: config file, default: configs/geme.yaml
## $2: which model to run, default: all!
##      options are mofa, moe, poe, mvib, cgae,

ml use /opt/insy/modulefiles;
ml load cuda/11.0;
ml load cudnn/11.0-8.0.3.33;


if [ $# -eq 0 ];
then
  CONFIG_FILE='configs/geme.yaml';
else
  CONFIG_FILE=$1;
fi

echo 'start';
TRAIN_LOG="train-log.txt";

if [ $# -eq 1 ];
then
  # no model specified, run all
  python run.py --experiment='experiment' --config=${CONFIG_FILE} >> ${TRAIN_LOG};
else

  python run.py --experiment='experiment' --config=${CONFIG_FILE} -$2 >> ${TRAIN_LOG};
fi;
