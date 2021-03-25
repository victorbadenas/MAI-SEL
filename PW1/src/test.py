import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from dataset import PandasDataset
# from prism import Prism
from rule_interpreter import RuleInterpreter


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
    parser.add_argument('-r', "--rules_path", type=Path, required=True)
    parser.add_argument('-i', "--dataset_path", type=Path, required=True)
    parser.add_argument('-d', "--debug", action="store_true", default=False)
    return parser.parse_args()


class Inference:
    def __init__(self, args):
        self.args = args
        self.show_parameters(args)
        self.rules_path = args.rules_path
        logging.info(f'loading data from {self.args.dataset_path}')
        self.dataset = PandasDataset(self.args.dataset_path)
        logging.info(f'data loaded')
        logging.debug(self.dataset)

    @staticmethod
    def show_parameters(parameters):
        logging.info("Parameters:")
        for label, value in parameters.__dict__.items():
            logging.info(f"\t{label}: {value}")

    def __call__(self):
        X = self.dataset.input_data
        Y = self.dataset.target_data

        logging.info(f'loading rules from {self.rules_path}')
        ri = RuleInterpreter(self.rules_path)
        logging.info(f'rules loaded successfully')

        logging.info('inferring')
        y = ri.predict(X)
        logging.info('infered!')

        accuracy = sum(1 for idx in range(len(y)) if y[idx] == Y[idx])/len(y)
        coverage = sum(1 for item in y if y is not None)/len(y)

        logging.info(f'Inference results for {self.dataset.dataset_path}')
        logging.info(f'accuracy: {accuracy*100:.2f}%')
        logging.info(f'coverage: {coverage*100:.2f}%')

if __name__ == "__main__":
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, debug=args.debug)
    Inference(args)()
