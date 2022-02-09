
CONFIGDIR=$1;
MODEL=$2;

if [ $MODEL == 'cgae' ];
then
  MODELNAME='CGAE';
elif [ $MODEL == 'poe' ];
then
  MODELNAME='PoE';
elif [ $MODEL == 'moe' ];
then
  MODELNAME='MoE';
else
  MODELNAME='MVIB';
fi;


RESPATH='/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/results/';


echo $RESNAME


for conf in $CONFIGDIR$MODEL'_'*'.yaml';
do
  if [[ $conf == *"default"* ]];
  then
    echo $conf;

  else

    RESNAME='train-tcga-'$MODEL'_'$VIEW1'_'$VIEW2'/'$MODELNAME'/finalValidationLoss.pkl';


    echo $conf $MODEL;
    #sbatch train_joint.sbatch $conf $MODEL;

  fi;
done;
