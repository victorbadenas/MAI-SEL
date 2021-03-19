import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from dataset import Dataset
from prism import Prism


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
    parser.add_argument('-m', "--conversion_mode", type=str, default='global')
    return parser.parse_args()


class Main:
    def __init__(self, args):
        self.args = args
        self.show_parameters(args)
        self.dataset = Dataset(self.args.dataset_path, mode=self.args.conversion_mode)
        logging.debug(self.dataset)

    @staticmethod
    def show_parameters(parameters):
        logging.info("Parameters:")
        for label, value in parameters.__dict__.items():
            logging.info(f"\t{label}: {value}")

    def __call__(self):
        Y = self.dataset['target']
        X = self.dataset.drop('target', axis=1)
        prism = Prism()
        prism.fit(X, Y)
        logging.debug('\n'.join(map(str, prism.rules)))


if __name__ == "__main__":
    args = parseArgumentsFromCommandLine()
    set_logger(args.logger, debug=args.debug)
    Main(args)()
