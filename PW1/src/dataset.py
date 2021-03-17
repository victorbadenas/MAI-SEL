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
        self.convertToNumerical()
        self.data.columns = self.data.columns.to_list()[:-1] + ['target']

    def checkArgs(self):
        assert self.datasetPath.exists(), f'File {self.datasetPath} does not exist'
        assert self.mode, f'Mode {mode} is not supported. Options: {MODES}'

    def loadDataset(self, datasetPath):
        return pd.read_csv(datasetPath, header=None)

    def convertToNumerical(self):
        if self.mode == GLOBAL:
            self.convertGlobal()
        elif self.mode == COLUMNS:
            self.convertPerColumn()
        self.convertLabels()

    def convertGlobal(self):
        self.labelEncoder = LabelEncoder()
        flattenedData = self.data[self.data.columns[:-1]].to_numpy().flatten()
        self.labelEncoder.fit(flattenedData)
        for columnName in self.data.columns[:-1]:
            self.data[columnName] = self.labelEncoder.transform(self.data[columnName])

    def convertLabels(self):
        self.targetEncoder = LabelEncoder()
        targetName = self.data.columns[-1]
        self.data[targetName] = self.targetEncoder.fit_transform(self.data[targetName])

    def convertPerColumn(self):
        self.encoders = dict()
        for columnName in self.data.columns[:-1]:
            self.encoders[columnName] = LabelEncoder()
            self.data[columnName] = self.encoders[columnName].fit_transform(self.data[columnName])

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
