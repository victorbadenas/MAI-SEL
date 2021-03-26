#!/bin/sh

rm -r build/ dist/ *.spec

cd src/
pyinstaller --hidden-import cmath train.py
pyinstaller --hidden-import cmath test.py
mv dist/ ../dist/
mv build/ ../build/
mv *.spec ../
cd -
