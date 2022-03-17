#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --time=04:00:00
#SBATCH --job-name=evalJoint
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL


CONFDIR='./configs/';
RESULTDIR='/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/results/';

EMBEDDINGSDIR='./embeddings/';


# create a new directory to save the config files to be tested
mkdir -p $CONFDIR'best-models/';
mkdir -p $EMBEDDINGSDIR;

# for each dataset, get the best hyperparameters for each model
# and then create the new config files for loading them
python src/util/bestModel.py $RESULTDIR $CONFDIR;



for dataset in 'RNAADT' 'GEME' 'GECNV' 'MECNV';
do
  echo $dataset;

  for model in 'baseline' 'cgae' 'mvib' 'moe' 'poe';
  do
    CONFIG=$CONFDIR'best-models/'$dataset'_'$model'.yaml';

    python run.py --experiment='experiment' --config=$CONFIG -$model --results-path=embeddings;
  done;
done;
