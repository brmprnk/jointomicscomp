#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=03:30:00
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
# python src/util/bestModel.py $RESULTDIR $CONFDIR;


#for dataset in 'RNAADT' 'GEME' 'GECNV' 'MECNV';
#for dataset in 'GEME' 'GECNV';
#for dataset in 'RNAADT';
#do
#  echo $dataset;

#  #for model in 'cgae' 'cvae' 'moe' 'poe';
#  for model in 'poe';
#  do
#    CONFIG=$CONFDIR'best-models/'$dataset'_'$model'.yaml';

#    python run.py --experiment='experiment' --config=$CONFIG -$model --results-path=embeddings;
#  done;

  #python run.py --experiment='experiment' --config='configs/sc/baseline.yaml' -baseline --results-path='embeddings';
#done;

#python run.py --experiment='experiment' --config='configs/tcga/GEME/baseline.yaml' -baseline --results-path='embeddings';
#python run.py --experiment='experiment' --config='configs/tcga/GECNV/baseline.yaml' -baseline --results-path='embeddings';
#python run.py --experiment='experiment' --config='configs/sc/baseline.yaml' -baseline --results-path='embeddings';
#python run.py --experiment='experiment' --config='configs/sc/baseline_l3.yaml' -baseline --results-path='embeddings';

#for model in 'poe';
dataset='RNAADT';
#
for model in 'cgae' 'cvae' 'moe' 'poe';
do
  rm 'embeddings/results/test_RNA_ADT_'$model'_RNA_ADT/' -r
  CONFIG=$CONFDIR'best-models/'$dataset'_'$model'_l3.yaml';

  python run.py --experiment='experiment' --config=$CONFIG -$model --results-path=embeddings;
done;
#
# rm 'embeddings/results/test-cite-baseline_RNA_ADT/' -r
# python run.py --experiment='experiment' --config='configs/sc/baseline_l3.yaml' -baseline --results-path='embeddings';
