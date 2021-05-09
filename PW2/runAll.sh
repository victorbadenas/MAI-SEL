#!/bin/sh

python scripts/train_all.py -i data/iris.train -i data/car.train -i data/kr-vs-kp.train -f models/ -t -1 -m RF
[ $? -ne 0 ] && exit 1
python scripts/train_all.py -i data/iris.train -i data/car.train -i data/kr-vs-kp.train -f models/ -t -1 -m DF
[ $? -ne 0 ] && exit 1
python scripts/test_all.py -i data/iris.test -i data/car.test -i data/kr-vs-kp.test -md models/ -t -1
[ $? -ne 0 ] && exit 1
