#! /bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=01:00:00
#SBATCH --job-name=evalJoint
#SBATCH --mail-user=stavrosmakrodi
#SBATCH --mail-type=ALL


CONFDIR='./configs/';
#RESULTDIR='/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/results/';

EMBEDDINGSDIR='./embeddings/';


# create a new directory to save the config files to be tested
mkdir -p $CONFDIR'best-models/';
mkdir -p $EMBEDDINGSDIR;

# for each dataset, get the best hyperparameters for each model
# and then create the new config files for loading them
# python src/util/bestModel.py $RESULTDIR $CONFDIR;


#for dataset in 'RNAADT' 'GEME' 'GECNV' 'MECNV';
#for dataset in 'GEME' 'GECNV';
for dataset in 'RNAADT';
do
  echo $dataset;

  #for model in 'cgae' 'cvae' 'moe' 'poe' 'totalvi' 'uniport';
  for model in 'uniport';
  do
    CONFIG=$CONFDIR'best-models/'$dataset'_'$model'.yaml';

    python run.py --experiment='experiment' --config=$CONFIG -$model --results-path=embeddings;
  done;

  #python run.py --experiment='experiment' --config='configs/sc/baseline.yaml' -baseline --results-path='embeddings';
done;

#python run.py --experiment='experiment' --config='configs/tcga/GEME/baseline.yaml' -baseline --results-path='embeddings';
#python run.py --experiment='experiment' --config='configs/tcga/GECNV/baseline.yaml' -baseline --results-path='embeddings';
#python run.py --experiment='experiment' --config='configs/sc/baseline.yaml' -baseline --results-path='embeddings';
#python run.py --experiment='experiment' --config='configs/sc/baseline_l3.yaml' -baseline --results-path='embeddings';
#exit;
#for model in 'poe';
#dataset='RNAATAC';
#
#for model in 'cgae' 'cvae' 'moe' 'poe';
#for model in 'cgae' 'cvae' 'moe';
#do
#  rm 'embeddings/results/test_RNA_ATAC_'$model'_RNA_ATAC/' -r
#  CONFIG=$CONFDIR'best-models/'$dataset'_'$model'.yaml';

#  python run.py --experiment='experiment' --config=$CONFIG -$model --results-path=embeddings;
#done;
#
#python run.py --experiment='experiment' --config=configs/best-models/GEME_uniport.yaml -uniport --results-path=embeddings;
#python run.py --experiment='experiment' --config=configs/best-models/GECNV_uniport.yaml -uniport --results-path=embeddings;
# rm 'embeddings/results/test-cite-baseline_RNA_ADT/' -r
#python run.py --experiment='experiment' --config='configs/best-models/RNAATAC_baseline.yaml' -baseline --results-path='embeddings/results/test_RNA_ATAC_baseline-l2_RNA_ATAC/';
