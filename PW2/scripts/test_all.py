import os
import sys
import time
import tqdm
import random
import argparse
import logging
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

sys.path.append('./src/')

from dataset import PandasDataset
from forest_interpreter import forest_from_json


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
    parser.add_argument('-l', "--logger", type=Path, default=f"log/{os.path.splitext(os.path.basename(__file__))[0]}.log")
    parser.add_argument('-i', "--dataset_paths", action='append', required=True)
    parser.add_argument('-d', "--debug", action="store_true", default=False)
    parser.add_argument('-t', "--num_threads", type=int, default=1, help='number of threads for RandomForestClassifier. -1 for using all threads as given by os.cpu_count(). default=1')
    parser.add_argument('-md', "--models_dir", type=Path, default='.models/')
    return parser.parse_args()


def main(args):
    for dataset_path in args.dataset_paths:
        dataset = PandasDataset(dataset_path)
        logging.info(dataset.name)
        fit_dataset(dataset, models_dir=args.models_dir, n_jobs=args.num_threads)


def fit_dataset(dataset, models_dir=None, n_jobs=1):
    X = dataset.input_data
    Y = dataset.target_data

    model_folders = list((models_dir / dataset.name).glob('*/'))

    (models_dir / dataset.name).mkdir(exist_ok=True, parents=True)
    csv_f = open(models_dir / dataset.name / 'test.csv', 'w')
    csv_f.write(f'name,predict_time(s),test_acc\n')

    for model_folder in tqdm.tqdm(model_folders):
        if not model_folder.is_dir():
            continue
        model_name = model_folder.parts[-1]
        model_json_path = model_folder / 'model.json'
        fi = forest_from_json(model_json_path, n_jobs=n_jobs)
        st = time.time()
        y = fi.predict(X)
        en = time.time() - st
        acc = (Y == y).mean()
        logging.info(f'model {model_folder}: acc={acc*100:.2f}% in {en:.2f}s')
        with open(model_folder / 'results.txt', 'a+') as f:
            f.write(f'Test accuracy: {acc*100:.2f}%')
        csv_f.write(f'{model_name},{en},{acc}\n')
    csv_f.close()

    # sort_by name
    pd.read_csv(models_dir / dataset.name / 'test.csv', index_col=0).sort_index().to_csv(models_dir / dataset.name / 'sorted_test.csv')

if __name__ == "__main__":
    set_seeds(42)
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, debug=args.debug)
    main(args)