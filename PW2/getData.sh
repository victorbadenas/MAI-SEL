#!/bin/sh

get_data () {
    url=$1
    headers=$2
    filename=data/$(basename ${url})
    echo ${headers} > ${filename}
    curl ${url} >> ${filename}
}

mkdir data/

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data 'buying,maint,doors,persons,lug_boot,safety,class'

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data 'A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16'

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data 'sepal_length,sepal_width,petal_length,petal_width,class'


for fullfile in data/*; do

    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"

    if [ "$extension" = "data" ]; then
        python scripts/split_train_test.py -i $fullfile
        rm $fullfile
    fi
done
