import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('src/')

from dataset import PandasDataset

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
    #     self.label_dist = self.get_data_distribution()

    # def get_data_distribution(self):
    #     labels = self.dataset.target_data
    #     labels_unique, labels_count = np.unique(labels, return_counts=True)
    #     labels_count = labels_count / len(labels)
    #     label_dist = dict(zip(labels_unique, labels_count))
    #     return label_dist

    def __call__(self, *args, **kwargs):
        self.split_dataset()
        if self.save:
            self.save_splits()

    def split_dataset(self):
        train_data = pd.DataFrame(columns=self.dataset.columns)
        test_data = pd.DataFrame(columns=self.dataset.columns)
        for cls in self.unique_labels:
            cls_data = self.dataset[self.dataset.target_data == cls]
            indexes = cls_data.index
            train_items_index = np.random.choice(indexes, int(self.args.train_perc*len(indexes)), replace=False)
            test_items_index = sorted(set(indexes) - set(train_items_index))
            train_data = pd.concat((train_data, cls_data.iloc[train_items_index]))
            test_data = pd.concat((train_data, cls_data.iloc[test_items_index]))
            pass

    def save_splits(self):
        pass


if __name__ == "__main__":
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, args.debug)
    show_parameters(args)
    TrainTestSplitter(args)()
