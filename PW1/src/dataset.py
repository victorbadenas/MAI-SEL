import pandas as pd
from sklearn.preprocessing import LabelEncoder

GLOBAL = 'global'
COLUMNS = 'columns'
MODES = [GLOBAL, COLUMNS]


class Dataset:
    def __init__(self, datasetPath, mode=GLOBAL):
        self.datasetPath = datasetPath
        self.mode = mode
        self.checkArgs()
        self.data = self.loadDataset(datasetPath)
        self.data.columns = self.data.columns.to_list()[:-1] + ['target']

    def checkArgs(self):
        assert self.datasetPath.exists(), f'File {self.datasetPath} does not exist'
        assert self.mode, f'Mode {mode} is not supported. Options: {MODES}'

    def loadDataset(self, datasetPath):
        return pd.read_csv(datasetPath, header=None)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def columns(self):
        return self.data.columns

    def drop(self, *args, **kwargs):
        return self.data.drop(*args, **kwargs)
