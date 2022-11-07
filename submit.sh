
CONFIGDIR=$1;
MODEL=$2;
DATASETNAME=$3;


if [ $MODEL == 'cgae' ];
then
  MODELNAME='CGAE';
elif [ $MODEL == 'cvae' ];
then
  MODELNAME='CVAE';
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

if [ ${#VIEW1} -eq 0 ];
then
  VIEW1="RNA";
fi;


echo $VIEW1;

VIEW2=$(cut -c 3- <<< $VIEWS);

if [ ${#VIEW2} -eq 0 ];
then
  VIEW2="ADT";
fi;

echo $VIEW2;
for conf in $CONFIGDIR$MODEL'_'*'.yaml';
do
  if [[ $conf == *"default"* ]];
  then
    echo 'Skipping: '$conf;

  else
    FIELDS=(${conf//_/ });
    NUMBER=(${FIELDS[1]//./ });
    RESNAME='likelihood-'$DATASETNAME'-'$MODEL'-'$NUMBER'_'$VIEW1'_'$VIEW2'/'$MODELNAME'/finalValidationLoss.pkl';

    if [ ! -f $RESPATH$RESNAME ];
    then
      #echo $conf $MODEL;
      rm -r $RESPATH'likelihood-'$DATASETNAME'-'$MODEL'-'$NUMBER'_'$VIEW1'_'$VIEW2'/';
      sbatch train_joint.sbatch $conf $MODEL;
    else
      echo 'Skipping, result exists: '$conf;
    fi;


  fi;
done;
