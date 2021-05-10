# PW2 Forest Classifiers

Repository for the PW2 practicum for MAI-SEL @ FiB-UPC

## Respository contents

- `data` folder with the datasets
- `doc` folder with papers and final latex report
- `log` folder with log files
- `models` output folder with output files from training
- `getData.sh` script to download the datasets
- `runAll.sh` script to train and test all the datasets
- `requirements.txt` python requirements file
- `scripts` folder with scripts
  - `create_csv_all.py` creates csv from the outputs from `test_all.py` and `train_all.py`
  - `test_all.py` infer for all the models in `models/`
  - `train_all.py` trains all the configurations of RF or DF
  - `split_train_test.py` script to create stratified test from UCI data files.
- `src` folder with the core algorithms
  - `base_classifier.py` contains `BaseClassifier`
  - `base_forest.py` contains `BaseForest`
  - `dataset.py` contains the dataset loading utilities
  - `decision_forest.py` contains `DecisionForestClassifier`
  - `forest_interpreter.py` contains `ForestInterpreter`
  - `__init__.py`
  - `random_forest.py` contains `RandomForestClassifier`
  - `tree_units.py` contains `Tree, Node, Leaf`
  - `utils` utility functions
    - `data.py`
    - `__init__.py`
    - `math.py`
    - `time.py`

## Respository structure

```text
.
├── data
│   ├── car.test
│   ├── car.train
│   ├── iris.test
│   ├── iris.train
│   ├── kr-vs-kp.test
│   └── kr-vs-kp.train
├── doc
│   ├── LBreiman- Random Forests- Machine Learning-45-1-2001.pdf
│   ├── PW2-SEL-2021.pdf
│   ├── report
│   │   └── ...
│   └── Tim Kam Ho-The Random Subspace Method for Constructing Decision Forests-IEEE Trans.on PAMI-20-8-1998.pdf
├── getData.sh
├── log
│   ├── test_all.log
│   └── train_all.log
├── models
│   ├── car
│   │   └── ...
│   ├── iris
│   │   └── ...
│   └── kr-vs-kp
│       └── ...
├── readme.md
├── requirements.txt
├── runAll.sh
├── scripts
│   ├── create_csv_all.py
│   ├── split_train_test.py
│   ├── test_all.py
│   └── train_all.py
└── src
    ├── base_classifier.py
    ├── base_forest.py
    ├── dataset.py
    ├── decision_forest.py
    ├── forest_interpreter.py
    ├── __init__.py
    ├── random_forest.py
    ├── tree_units.py
    └── utils
        ├── data.py
        ├── __init__.py
        ├── math.py
        └── time.py
```

## Execution instuctions

### Linux

Envionment install with conda:

```bash
conda create --name sel3.9 python=3.9
conda activate sel3.9
pip install -r requirements.txt
```

To get the data in the right format:

```bash
./getData.sh
```

To run all the training and testing:

```bash
./runAll.sh
```

### Other OS

Environment install with conda:

```bash
conda create --name sel3.9 python=3.9
conda activate sel3.9
pip install -r requirements.txt
```

if you are on a UNIX based system with bash you can run `./getData.sh` from CLI or follow the following steps:

- Download the datasets from UCI repository:
  - <https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data>
  - <https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data>
  - <https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data>
- move them to the `data/` folder
- manually add the headers for the datasets:
  - kr-vs-kp.data: bkblk,bknwy,bkon8,bkona,bkspr,bkxbq,bkxcr,bkxwp,blxwp,bxqsq,cntxt,dsopp,dwipd,hdchk,katri,mulch,qxmsq,r2ar8,reskd,reskr,rimmx,rkxwp,rxmsq,simpl,skach,skewr,skrxp,spcop,stlmt,thrsk,wkcti,wkna8,wknck,wkovl,wkpos,wtoeg,class
  - car.data: buying,maint,doors,persons,lug_boot,safety,class
  - iris.data: sepal_length,sepal_width,petal_length,petal_width,class
- split the data into train and test with the following command:

```bash
python scripts/split_train_test.py -i data/car.data
python scripts/split_train_test.py -i data/kr-vs-kp.data
python scripts/split_train_test.py -i data/iris.data
```

Once the data is formatted correctly into the `data/` subfolder, run the training and inference with the following commands:

```bash
# Train random forest
python scripts/train_all.py \
    -i data/iris.train \
    -i data/car.train \
    -i data/kr-vs-kp.train \
    -f models/ \
    -t -1 \
    -m RF
# Train Decision forest
python scripts/train_all.py \
    -i data/iris.train \
    -i data/car.train \
    -i data/kr-vs-kp.train \
    -f models/ \
    -t -1 \
    -m DF
# Test all the datasets
python scripts/test_all.py \
    -i data/iris.test \
    -i data/car.test \
    -i data/kr-vs-kp.test \
    -md models/ \
    -t -1
```

finally to build the csvs of the report:

```bash
python scripts/create_csv_all.py
```

In the end, the models will be saved to `models/dataset/<model_config>/{model.json|feat_counts.json|model.trees|results.txt}`. Where the `model.json` contains the serialized object to be loaded by `ForestInterpreter`, the `feat_counts.json` file contains the occurrences of each of the features in a node. The `results.txt` contains the train and test accuracies for the model. The `model.trees` contains a text representation of the trees for that model. f.i:

```text
Tree 0:
(safety == low)
├t─ {'unacc': 1.0}
└f─ (safety == high)
    ├t─ {'unacc': 0.48, 'acc': 0.36, 'vgood': 0.11, 'good': 0.05}
    └f─ {'unacc': 0.61, 'acc': 0.31, 'good': 0.07}
```
