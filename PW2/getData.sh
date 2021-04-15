#!/bin/sh

get_data () {
    url=$1
    headers=$2
    filename=data/$(basename ${url})
    echo ${headers} > ${filename}
    curl ${url} >> ${filename}
}

mkdir data/


get_data https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data 'sepal_length,sepal_width,petal_length,petal_width,class'

for fullfile in data/*; do

    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"

    if [ "$extension" = "data" ]; then
        python scripts/split_train_test.py -i $fullfile
    fi
done
