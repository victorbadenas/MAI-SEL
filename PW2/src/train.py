import os
import sys
import random
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from dataset import PandasDataset
from decision_forest import DecisionForestClassifier
from random_forest import RandomForestClassifier

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

def set_logger(log_file_path, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)


def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser("")
    parser.add_argument('-l', "--logger", type=Path, default="log/log.log")
    parser.add_argument('-i', "--dataset_path", type=Path, required=True)
    parser.add_argument('-d', "--debug", action="store_true", default=False)
    parser.add_argument('-f', "--output_dir", type=Path, default=None)
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.show_parameters(args)
        self.output_dir = self.args.output_dir
        self.dataset = PandasDataset(self.args.dataset_path)
        logging.debug(self.dataset)

    @staticmethod
    def show_parameters(parameters):
        logging.info("Parameters:")
        for label, value in parameters.__dict__.items():
            logging.info(f"\t{label}: {value}")

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)

    def main(self):
        X = self.dataset.input_data
        Y = self.dataset.target_data

        rfc = RandomForestClassifier(F=2, num_trees=3, n_jobs=20)

        rfc.fit(X, Y)

        logging.debug(f'\n{rfc}')

        # accuracy = sum(1 for idx in range(len(y)) if y[idx] == Y[idx])/len(y)
        # coverage = sum(1 for item in y if y is not None)/len(y)

        # logging.info(f'Inference results on training data for {self.dataset.dataset_path}')
        # logging.info(f'accuracy: {accuracy*100:.2f}%')
        # logging.info(f'coverage: {coverage*100:.2f}%')

        if self.output_dir is not None:
            # json_model_path = self.output_dir / (self.dataset.name + '.json')
            tree_model_path = self.output_dir / (self.dataset.name + '.trees')
            # logging.info(f"Saving json model to {json_model_path}")
            # rfc.save(json_model_path)
            logging.info(f"Saving tree to {tree_model_path}")
            rfc.save(tree_model_path)

if __name__ == "__main__":
    # set_seeds(42)
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, debug=args.debug)
    Trainer(args)()
