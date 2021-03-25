#!/bin/sh

get_data () {
    url=$1
    headers=$2
    filename=data/$(basename ${url})
    echo ${headers} > ${filename}
    curl ${url} >> ${filename}
}

mkdir data/

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data '0,1,2,3,4,5,6,7,8,target'

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.data 'name,hobby,age,educational level,marital status,class'
python -c "import pandas as pd; data=pd.read_csv('data/hayes-roth.data'); data=data.drop('name', axis=1); data.to_csv('data/hayes-roth.train', index=False)"
rm data/hayes-roth.data

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.test 'hobby,age,educational level,marital status,class'

get_data https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data 'bkblk,bknwy,bkon8,bkona,bkspr,bkxbq,bkxcr,bkxwp,blxwp,bxqsq,cntxt,dsopp,dwipd,hdchk,katri,mulch,qxmsq,r2ar8,reskd,reskr,rimmx,rkxwp,rxmsq,simpl,skach,skewr,skrxp,spcop,stlmt,thrsk,wkcti,wkna8,wknck,wkovl,wkpos,wtoeg,class'
