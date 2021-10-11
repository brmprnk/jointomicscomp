#!/bin/bash

cd `dirname "$0"`
cd "data"

test -e 'cancer' || mkdir cancer
cd cancer

for cancer in breast lung
do
	test -e "$cancer" && continue
	mkdir "$cancer"
	cd "$cancer"
	curl -O "http://acgt.cs.tau.ac.il/multi_omic_benchmark/data/$cancer.zip"
	unzip "$cancer.zip"
	rm "$cancer.zip"
	cd ..
done

cd ..
if [ ! -e "clinical" ]
then
	curl -O http://acgt.cs.tau.ac.il/multi_omic_benchmark/data/clinical.zip
	unzip clinical.zip
	rm clinical.zip
fi
