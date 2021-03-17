#!/bin/sh

mainFile=${1:-"src/main.py"}

rm -r build/ dist/ main.spec

pyinstaller --onefile $mainFile
