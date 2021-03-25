#!/bin/sh

rm -r build/ dist/ train.spec test.spec

cd src/
pyinstaller --hidden-import cmath train.py
pyinstaller --hidden-import cmath test.py
mv dist/ ../dist/
mv build/ ../build/
mv train.spec ../train.spec
mv test.spec ../test.spec
cd -
