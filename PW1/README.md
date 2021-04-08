# PW1-SEL-2021-VictorBadenas

PRISM implementation of the paper in `doc/PRISM-Int.J.Man-Machine Studies-27-349-370-1987.pdf`

## Directory structure

The root folder contains:

- `build` folder with the linux binaries.
- `build.sh` script for building the project in Linux.
- `data` folder with the datasets.
- `dist` folder with the binaries created in the pyinstaller compilation.
- `doc` folder with the enunciate of the PW1 work, the paper on PRISM and the report in Latex and pdf format.
- `getData.sh` script to automatically download and format the data in linux.
- `log` folder with the train and test log files
- `models` folder with the model json objects and the rules in a `.rules` file in readable format.
- `README.md`: this file.
- `requirements.txt` file with the python dependancies of the project
- `scripts` folder with the `split_train_test.py` script
- `src` folder with the code of the train and test algorithms in the project.
  - `dataset.py`: contains the dataset object to read and preprocess a csv file
  - `prism.py`: contains the prism python object and it's methods
  - `rule_interpreter.py`: main inferencer object reading rules in the format created by prism
  - `rule.py`: contains the rule object and all of it's methods
  - `test.py`: main test script. calls rule_interpreter internally
  - `train.py`: main train script. calls prism internally
- `test.sh`: scipt to run all inferences in linux for the datasets in `data/` with .test extension.
- `train.sh`: script to run all trainings in linux for the datasets in `data/` with the .train extension.

```text
├── build
│   ├── split_train_test
│   │   └── split_train_test
│   ├── test
│   │   └── test
│   └── train
│       └── train
├── build.sh
├── data
│   ├── car.data
│   ├── car.test
│   ├── car.train
│   ├── hayes-roth.test
│   ├── hayes-roth.train
│   ├── kr-vs-kp.data
│   ├── kr-vs-kp.test
│   └── kr-vs-kp.train
├── dist
│   └── ...
├── doc
│   ├── PRISM-Int.J.Man-Machine Studies-27-349-370-1987.pdf
│   ├── PW1-Report-Victor-Badenas.pdf
│   ├── PW1-SEL-2021.pdf
│   └── report
│       └── ...
├── getData.sh
├── log
│   ├── car.train.log
│   ├── hayes-roth.train.log
│   ├── kr-vs-kp.train.log
│   ├── log.log
│   └── test.log
├── models
│   ├── car.json
│   ├── car.rules
│   ├── hayes-roth.json
│   ├── hayes-roth.rules
│   ├── kr-vs-kp.json
│   └── kr-vs-kp.rules
├── README.md
├── requirements.txt
├── scripts
│   └── split_train_test.py
├── src
│   ├── dataset.py
│   ├── prism.py
│   ├── rule_interpreter.py
│   ├── rule.py
│   ├── test.py
│   └── train.py
├── test.sh
└── train.sh
```

## Linux Data instructions

Datasets are retrieved from [UCI-ML Datasets Repository](https://archive.ics.uci.edu/ml/datasets.php?format=&task=&att=cat&area=&numAtt=&numIns=&type=&sort=nameUp&view=table)

```bash
./getData.sh
```

## Other OS data

create python environment

```bash
conda create --name sel3.9 python=3.9
conda activate sel3.9
pip install -r requirements.txt
```

Download data from the following urls and add it to the data folder:

- https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.test
- https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data

Filter unused column in hayes-roth dataset:

```bash
python -c "import pandas as pd; data=pd.read_csv('data/hayes-roth.data'); data=data.drop('name', axis=1); data.to_csv('data/hayes-roth.train', index=False)"
```

remove hayes-roth.data dataset

Add the headers for each dataset:

- hayes-roth.train: hobby,age,educational level,marital status,class
- hayes-roth.test: hobby,age,educational level,marital status,class
- kr-vs-kp.data: bkblk,bknwy,bkon8,bkona,bkspr,bkxbq,bkxcr,bkxwp,blxwp,bxqsq,cntxt,dsopp,dwipd,hdchk,katri,mulch,qxmsq,r2ar8,reskd,reskr,rimmx,rkxwp,rxmsq,simpl,skach,skewr,skrxp,spcop,stlmt,thrsk,wkcti,wkna8,wknck,wkovl,wkpos,wtoeg,class
- car.data: buying,maint,doors,persons,lug_boot,safety,class

create the splits:

```python
python scripts/split_train_test.py -i data/car.data
python scripts/split_train_test.py -i data/kr-vs-kp.data
```

## for Linux

### train

run the bash script that calls the executables

```bash
./train.sh
```

### test

run the bash script that calls the executables

```bash
./test.sh
```

## Other OS execution

### train

call the python scripts with the previously created environment

```bash
python src/train.py -i data/car.train -f models/ -l log/car.train.log
python src/train.py -i data/hayes-roth.train -f models/ -l log/hayes-roth.train.log
python src/train.py -i data/kr-vs-kp.train -f models/ -l log/kr-vs-kp.train.log
```

### test

call the python scripts with the previously created environment

```bash
python src/test.py -i data/car.test -r models/car.rules
python src/test.py -i data/hayes-roth.test -r models/hayes-roth.rules
python src/test.py -i data/kr-vs-kp.test -r models/kr-vs-kp.rules
```

## compilation

To compile the project, a bash script `./build.sh` is provided and it is imperative to have pyinstaller in the python environment.

### Linux, MacOS and UNIX systems

```bash
./build.sh
```

### Windows **NOT TESTED**

from the src folder

```powershell
pyinstaller --hidden-import cmath train.py
pyinstaller --hidden-import cmath test.py
```

move the dist and build folder to the PW1 directory, as well as the spec file to the same location.

from the scripts folder:

```powershell
pyinstaller --hidden-import cmath split_train_test.py
```

move the dist/split_train_test to the dist folder in the PW1 directory alongside the train and test folders. Same for the build and the spec file.

The binary can be executed the same way as it is in unix systems:

```powershell
.\dist\split_train_test\split_train_test -i path\to\file
```

```powershell
.\dist\train\train -i path\to\file -f models\ -l log\<logname>.log
```

```powershell
.\dist\test\test -i path\to\file -f models\<filename>.rules
```
