#!/bin/sh

for fullfile in data/*; do

    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"

    if [ $extension != "test" ]; then
        ./dist/train/train -i $fullfile -f models/ -l log/$filename.$extension.log
    fi
done
