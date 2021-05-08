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

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data 'bkblk,bknwy,bkon8,bkona,bkspr,bkxbq,bkxcr,bkxwp,blxwp,bxqsq,cntxt,dsopp,dwipd,hdchk,katri,mulch,qxmsq,r2ar8,reskd,reskr,rimmx,rkxwp,rxmsq,simpl,skach,skewr,skrxp,spcop,stlmt,thrsk,wkcti,wkna8,wknck,wkovl,wkpos,wtoeg,class'

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
