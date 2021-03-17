#!/bin/sh

mainFile=${1:-"main.py"}

rm -r build/ dist/ main.spec

cd src/
pyinstaller --hidden-import cmath $mainFile
mv dist/ ../dist/
mv build/ ../build/
mv main.spec ../main.spec
cd -
