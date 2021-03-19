import pandas as pd

class Dataset:
    def __init__(self, datasetPath):
        self.datasetPath = datasetPath
        self.checkArgs()
        self.data = self.loadDataset(datasetPath)
        self.targetLabel = self.columns[-1]

    def checkArgs(self):
        assert self.datasetPath.exists(), f'File {self.datasetPath} does not exist'

    def loadDataset(self, datasetPath):
        return pd.read_csv(datasetPath)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        return self.data[index]

    def drop(self, *args, **kwargs):
        return self.data.drop(*args, **kwargs)

    @property
    def columns(self):
        return self.data.columns

    @property
    def inputData(self):
        return self.data.drop(self.targetLabel, axis=1)

    @property
    def targetData(self):
        return self.data[self.targetLabel]

    @property
    def name(self):
        return self.datasetPath.stem