import os
import sys
import time
import random
import argparse
import logging
import numpy as np
from itertools import product
from pathlib import Path

sys.path.append('./src/')

from dataset import PandasDataset
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
    parser.add_argument('-l', "--logger", type=Path, default="log/all_rf.log")
    parser.add_argument('-i', "--dataset_paths", action='append', required=True)
    parser.add_argument('-d', "--debug", action="store_true", default=False)
    parser.add_argument('-f', "--output_dir", type=Path, required=True)
    return parser.parse_args()

def main(args):
    for dataset_path in args.dataset_paths:
        dataset = PandasDataset(dataset_path)
        logging.info(dataset.name)
        fit_dataset(dataset, output_dir=args.output_dir)

def fit_dataset(dataset, output_dir=None):
    X = dataset.input_data
    Y = dataset.target_data
    n_features = len(dataset.columns)-1

    Fs = set([1, 3, int(np.log2(n_features + 1)), int(np.sqrt(n_features))])
    NTs = 1, 10, 25, 50, 75, 100

    for F, NT in product(Fs, NTs):
        logging.info(f"fitting for F={F}, NT={NT}")
        rfc = RandomForestClassifier(F=F, num_trees=NT, n_jobs=-1, classKey=Y.name)
        st = time.time()
        rfc.fit(X, Y)
        logging.info(f"model fitted in {time.time() - st:.2f}s")
        if output_dir:
            logging.info(f'Saving...')
            save_model(rfc, dataset.name, output_dir, F=F, NT=NT)

def save_model(model, dataset_name, output_dir, **parameters):
    exp_folder = 'RF_' + '_'.join(f'{k}-{v}' for k,v in parameters.items())
    json_model_path = output_dir / dataset_name / exp_folder / 'model.json'
    tree_model_path = output_dir / dataset_name / exp_folder / 'model.trees'
    (output_dir / dataset_name / exp_folder).mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving json model to {json_model_path}")
    model.save(json_model_path)
    logging.info(f"Saving tree to {tree_model_path}")
    model.save(tree_model_path)

if __name__ == "__main__":
    set_seeds(42)
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, debug=args.debug)
    main(args)