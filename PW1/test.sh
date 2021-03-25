#!/bin/sh

for fullfile in data/*; do

    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"

    if [ "$extension" = "test" ]; then
        ./dist/test/test -i $fullfile -r models/$filename.rules
    fi
done