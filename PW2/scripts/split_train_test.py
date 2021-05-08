import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('src/')

from dataset import PandasDataset

np.random.seed(42)

def set_logger(log_file_path, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)


def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser("")
    parser.add_argument('-l', "--logger", type=Path, default=None)
    parser.add_argument('-i', "--dataset_path", type=Path, required=True)
    parser.add_argument('-d', "--debug", action="store_true", default=False)
    parser.add_argument('-t', "--train_perc", default=.7, help='percentage of values assigned to train')
    return parser.parse_args()

def show_parameters(parameters):
    logging.info("Parameters:")
    for label, value in parameters.__dict__.items():
        logging.info(f"\t{label}: {value}")

class TrainTestSplitter:
    def __init__(self, args, save=True):
        self.args = args
        self.save = save
        self.dataset = PandasDataset(self.args.dataset_path)
        logging.debug(self.dataset)
        self.unique_labels = self.dataset.target_data.unique()

    def __call__(self, *args, **kwargs):
        train_data, test_data = self.split_dataset()
        if self.save:
            self.save_splits(train_data, test_data)

    def split_dataset(self):
        logging.info(f'Splitting {self.dataset.dataset_path} data')
        train_data = pd.DataFrame(columns=self.dataset.columns)
        test_data = pd.DataFrame(columns=self.dataset.columns)

        for class_ in self.unique_labels:
            logging.info(f'extracting splits for class {class_}')
            cls_data = self.dataset[self.dataset.target_data == class_]
            indexes = cls_data.index
            train_items_index = np.random.choice(len(indexes), int(self.args.train_perc*len(indexes)), replace=False)
            test_items_index = sorted(set(range(len(indexes))) - set(train_items_index))
            train_data = pd.concat((train_data, cls_data.iloc[train_items_index]))
            test_data = pd.concat((test_data, cls_data.iloc[test_items_index]))

        train_data.sort_index(inplace=True)
        test_data.sort_index(inplace=True)
        logging.info(f'obtained splits: train->{train_data.shape[0]}, test->{test_data.shape[0]}')
        return train_data, test_data

    def save_splits(self, train_data, test_data):
        train_path = self.dataset.dataset_path.with_suffix('.train')
        test_path = self.dataset.dataset_path.with_suffix('.test')
        logging.info(f'saving files to {train_path} and {test_path}')
        self.save_csv(train_data, train_path)
        self.save_csv(test_data, test_path)

    @staticmethod
    def save_csv(df:pd.DataFrame, path:Path):
        if path.exists():
            logging.warning(f'overwriting existing file {path}')
        df.to_csv(path, index=False)


if __name__ == "__main__":
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, args.debug)
    show_parameters(args)
    TrainTestSplitter(args)()
