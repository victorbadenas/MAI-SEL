import logging
import traceback
import pandas as pd
from pathlib import Path


class BaseDataset:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.check_args()
        self.check_abstract()

    def check_args(self):
        assert self.dataset_path.exists(), f'File {self.dataset_path} does not exist'

    def check_abstract(self):
        try:
            self.input_data
            self.target_data
        except NotImplementedError as e:
            logging.error(traceback.format_exc(e))
        except Exception as e:
            return

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def columns(self):
        return self.columns_

    @property
    def input_data(self):
        raise NotImplementedError('input_data is abstract property method, implement when inheriting')

    @property
    def target_data(self):
        raise NotImplementedError('target_data is abstract property method, implement when inheriting')

    @property
    def name(self):
        return self.dataset_path.stem


class PandasDataset(BaseDataset):
    def __init__(self, dataset_path):
        super(PandasDataset, self).__init__(dataset_path)
        self.data = self.load_dataset(dataset_path)
        # self.data = self.data.astype(str)
        self.data.columns = list(map(lambda x: x.replace(' ', '_'), self.data.columns))
        self.columns_ = self.data.columns
        self.target_label = self.columns[-1]

    def load_dataset(self, dataset_path):
        return pd.read_csv(dataset_path)

    @property
    def input_data(self):
        return self.data.drop(self.target_label, axis=1)

    @property
    def target_data(self):
        return self.data[self.target_label]


class ListDataset(BaseDataset):
    def __init__(self, dataset_path, header=True):
        super(ListDataset, self).__init__(dataset_path)
        self.data, self.columns_ = self.read_csv_data(dataset_path)
        self.target_label = self.columns[-1]

    def read_csv_data(self, dataset_path, header=True):
        with open(dataset_path, 'r') as f:
            data = list(map(lambda x: x.strip().split(','), f.readlines()))
        if header:
            header = data[0]
            data = data[1:]
        else:
            header = list(range(len(data[0])))
        return data, header

    @property
    def input_data(self):
        return [item[:-1] for item in self.data]

    @property
    def target_data(self):
        return [item[-1] for item in self.data]

    def __str__(self):
        str_repr = str(self.header) + '\n'
        str_repr += '\n'.join(map(str, self.data))
        return str_repr
