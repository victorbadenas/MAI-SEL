import copy
import json
import logging
import multiprocessing as mp
import random
import traceback

import numpy as np
import pandas as pd

from base_classifier import BaseClassifier
from forest_interpreter import ForestInterpreter
from tree_units import Tree, Node, Leaf
from utils.data import filterNone

class BaseForestClassifier(ForestInterpreter, BaseClassifier):
    def __init__(self, headers=None, F=-1, num_trees=10, classKey=None, n_jobs=1):
        super(BaseForestClassifier, self).__init__(headers)
        self.class_name = self.__class__.__name__
        self.classKey = classKey
        self.F = F
        self.num_trees = num_trees
        self.n_jobs = None if n_jobs < 1 else n_jobs

    def load(self, path_to_file):
        with open(path_to_file, 'r') as f:
            d = json.load(f)
        try:
            self.load_dict(d)
        except Exception as e:
            msg = 'could not load model:\n' + traceback.format_exc(e)
            logging.error(msg)
            raise e

    def get_feature_importance(self, sort=True):
        if self.trees is None:
            logging.error('Model not fitted')
            raise ValueError('Model not fitted')
        counts = dict(zip(self._attributes, [0]*len(self._attributes)))
        for tree in self.trees:
            if not tree.trained:
                continue
            tree_counts = tree.get_feature_importance(sort=False)
            for k in counts:
                if k in tree_counts:
                    counts[k] += tree_counts[k]
        if sort:
            counts = dict(sorted(list(counts.items()), key=lambda x: x[1], reverse=True))
        return counts

    def _reset(self):
        self.trees = None

    def _fit_tree(self, tree_idx, X):
        raise NotImplementedError('abstract method')

    def _init_trees(self):
        raise NotImplementedError('abstract method')

    def _fit(self, X):
        self._attributes = list(X.columns)
        self._attributes.remove(self.classKey)
        self._labels = list(X[self.classKey].unique())

        self.trees = [None] * self.num_trees

        if self.n_jobs == 1:
            for idx in range(self.num_trees):
                self.trees[idx] = self._fit_tree(idx, X)
        else:
            from functools import partial
            with mp.Pool(self.n_jobs) as p:
                self.trees = p.map(partial(self._fit_tree, X=X), range(self.num_trees))

        # filter trees that are none
        self.trees = filterNone(self.trees)

        if not all(t.trained for t in self.trees):
            fitted = sum(t.trained for t in self.trees)
            logging.warning(f'Not all the trees have been fitted. {len(self.trees) - fitted} of {len(self.trees)} have not been fitted')

    def _validate_train_data(self, X, Y=None):
        if isinstance(X, pd.DataFrame):
            if isinstance(Y, pd.Series):
                assert X.shape[0] == Y.shape[0], 'instance numbers inconsistent'
                if self.classKey is None:
                    self.classKey = Y.name
                return X.join(Y)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _save_to_json(self, path_to_file):
        if self.trees is None:
            logging.error('fit the model first')
            raise ValueError('fit the model first')
        jsondata = copy.deepcopy(self.__dict__)

        for i, tree in enumerate(jsondata['trees']):
            if not tree.trained:
                jsondata['trees'][i] = None
                continue
            jsondata['trees'][i] = tree.to_dict()
        jsondata['trees'] = filterNone(jsondata['trees'])
        with open(path_to_file, 'w') as f:
            json.dump(jsondata, f, indent=4)

    def _save_to_txt(self, path_to_file):
        msg = str(self)
        with open(path_to_file, 'w') as f:
            f.write(msg)

    def __str__(self):
        return '\n\n'.join(f'Tree {i}:\n{tree}' for i, tree in enumerate(self.trees))

    def __repr__(self):
        return str(self)
