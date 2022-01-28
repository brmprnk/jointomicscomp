
CONFIGDIR=$1;
MODEL=$2;


for conf in $CONFIGDIR$MODEL'_'*'.yaml';
do
  if [[ $conf == *"default"* ]];
  then
    echo $conf;

  else
    sbatch train_joint.sbatch $conf $MODEL;

  fi;
done;
