#!/bin/sh

mkdir data/

echo '0,1,2,3,4,5,6,7,8,target' > data/tic-tac-toe.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data -O data/tmp.csv
cat data/tmp.csv >> data/tic-tac-toe.data
rm data/tmp.csv

echo 'name,hobby,age,educational level,marital status,class' > data/hayes-roth.train
wget https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.data -O data/tmp.csv
cat data/tmp.csv >> data/hayes-roth.train
rm data/tmp.csv

# remove name from csv
python -c "import pandas as pd; data=pd.read_csv('data/hayes-roth.train'); data=data.drop('name', axis=1); data.to_csv('data/hayes-roth.train', index=False)"

echo 'hobby,age,educational level,marital status,class' > data/hayes-roth.test
wget https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.test -O data/tmp.csv
cat data/tmp.csv >> data/hayes-roth.test
rm data/tmp.csv

echo 'bkblk,bknwy,bkon8,bkona,bkspr,bkxbq,bkxcr,bkxwp,blxwp,bxqsq,cntxt,dsopp,dwipd,hdchk,katri,mulch,qxmsq,r2ar8,reskd,reskr,rimmx,rkxwp,rxmsq,simpl,skach,skewr,skrxp,spcop,stlmt,thrsk,wkcti,wkna8,wknck,wkovl,wkpos,wtoeg,class' > data/kr-vs-kp.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data -O data/tmp.csv
cat data/tmp.csv >> data/kr-vs-kp.data
rm data/tmp.csv
