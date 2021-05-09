import os
import sys
import tqdm
import time
import json
import random
import argparse
import logging
import numpy as np
from itertools import product
from pathlib import Path

sys.path.append('./src/')

from dataset import PandasDataset
from random_forest import RandomForestClassifier
from decision_forest import DecisionForestClassifier

MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "DecisionForestClassifier": DecisionForestClassifier,
    "RF": RandomForestClassifier,
    "DF": DecisionForestClassifier
}

MODEL_F = {
    "RandomForestClassifier": lambda i: set([1, 3, int(np.log2(i + 1)), int(np.sqrt(i))]),
    "DecisionForestClassifier": lambda i: list(set([int(.25*i), int(.5*i), int(.75*i)])) + ['random'],
    "RF": lambda i: set([1, 3, int(np.log2(i + 1)), int(np.sqrt(i))]),
    "DF": lambda i: list(set([int(.25*i), int(.5*i), int(.75*i)])) + ['random']
}

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
    parser.add_argument('-f', "--output_dir", type=Path, required=True)
    parser.add_argument('-d', "--debug", action="store_true", default=False)
    parser.add_argument('-m', "--model", type=str, default="RF")
    parser.add_argument('-t', "--num_threads", type=int, default=1, help='number of threads for RandomForestClassifier. -1 for using all threads as given by os.cpu_count(). default=1')
    return parser.parse_args()

def main(args):
    for dataset_path in args.dataset_paths:
        dataset = PandasDataset(dataset_path)
        logging.info(dataset.name)
        fit_dataset(dataset,
                    output_dir=args.output_dir,
                    n_threads=args.num_threads,
                    model_name=args.model)

def fit_dataset(dataset, output_dir=None, n_threads=1, model_name=""):
    X = dataset.input_data
    Y = dataset.target_data
    n_features = len(dataset.columns)-1

    # Fs = set([1, 3, int(np.log2(n_features + 1)), int(np.sqrt(n_features))])
    Fs = MODEL_F[model_name](n_features)
    NTs = 1, 10, 25, 50, 75, 100

    (output_dir / dataset.name).mkdir(exist_ok=True, parents=True)
    csv_f = open(output_dir / dataset.name / f'{model_name}_train.csv', 'w')
    csv_f.write(f'name,F,NT,fit_time(s),train_acc\n')

    for F, NT in tqdm.tqdm(product(Fs, NTs), total=len(Fs) * len(NTs)):
        logging.info(f"fitting for F={F}, NT={NT}")
        rfc = MODELS[model_name](F=F, num_trees=NT, n_jobs=n_threads, classKey=Y.name)
        st = time.time()
        rfc.fit(X, Y)
        en = time.time() - st
        y = rfc.predict(X)
        acc = (Y == y).mean()
        logging.info(f"model fitted in {en:.2f}s with train accuracy {acc*100:.2f}%")
        if output_dir:
            logging.info(f'Saving...')
            csv_f.write(f'{model_name},{F},{NT},{en},{acc}\n')
            save_model(rfc, model_name, dataset.name, output_dir, acc, F=F, NT=NT)
    csv_f.close()

def save_model(model, model_name, dataset_name, output_dir, acc, **parameters):
    exp_folder = f'{model_name}_' + '_'.join(f'{k}-{v}' for k,v in parameters.items())
    json_model_path = output_dir / dataset_name / exp_folder / 'model.json'
    tree_model_path = output_dir / dataset_name / exp_folder / 'model.trees'
    results_path = output_dir / dataset_name / exp_folder / 'results.txt'
    counts_path = output_dir / dataset_name / exp_folder / 'feat_counts.json'

    (output_dir / dataset_name / exp_folder).mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving json model to {json_model_path}")
    model.save(json_model_path)
    logging.info(f"Saving tree to {tree_model_path}")
    model.save(tree_model_path)

    feat_counts = model.get_feature_importance()
    with open(counts_path, 'w') as f:
        json.dump(feat_counts, f, indent=4)
    logging.info(f"node feature counts: {feat_counts}")

    with open(results_path, 'w') as f:
        f.write(f'Train accuracy: {acc*100:.2f}%')

if __name__ == "__main__":
    set_seeds(42)
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, debug=args.debug)
    main(args)