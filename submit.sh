
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

VVIEWS=(${CONFIGDIR//\// });
VIEWS=${VVIEWS[2]};

VIEW1=$(cut -c -2 <<< $VIEWS);
echo $VIEW1;

VIEW2=$(cut -c 3- <<< $VIEWS);
echo $VIEW2;


for conf in $CONFIGDIR$MODEL'_'*'.yaml';
do
  if [[ $conf == *"default"* ]];
  then
    echo 'Skipping: '$conf;

  else
    FIELDS=(${conf//_/ });
    NUMBER=$(cut -c -1 <<< ${FIELDS[1]});
    RESNAME='train-tcga-'$MODEL'-'$NUMBER'_'$VIEW1'_'$VIEW2'/'$MODELNAME'/finalValidationLoss.pkl';

    if [ ! -f $RESPATH$RESNAME ];
    then
      #echo $conf $MODEL;
      sbatch train_joint.sbatch $conf $MODEL;
    else
      echo 'Skipping, result exists: '$conf;
    fi;


  fi;
done;
