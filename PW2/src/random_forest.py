import copy
import json
import logging
import random
from pprint import pformat, pprint

import numpy as np
import pandas as pd

from base_classifier import BaseClassifier
from utils.math import gini_index, delta_gini
from utils.data import is_numeric


class RandomForestClassifier(BaseClassifier):
    def __init__(self, headers=None, F=-1, classKey=None):
        super(RandomForestClassifier, self).__init__(headers)
        self.classKey = classKey
        self.mainNode = None
        self.F = F

    def _reset(self):
        pass

    def load(self, path_to_file):
        pass

    def _fit(self, data):
        self._attributes = list(data.columns)
        self._attributes.remove(self.classKey)
        self._labels = list(data[self.classKey].unique())
        self.mainNode = Tree(self._attributes, self._labels, classKey=self.classKey, F=self.F)
        self.mainNode.fit(data)

    def _predict(self, X):
        pass

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
        # jsondata = copy.deepcopy(self.__dict__)
        # jsondata['mainNode'] = jsondata['mainNode'].to_dict()
        # with open(path_to_file, 'w') as f:
        #     json.dump(jsondata, f, indent=4)
        pass

    def _save_to_txt(self, path_to_file):
        msg = str(self)
        with open(path_to_file, 'w') as f:
            f.write(msg)

    def __str__(self):
        return self.mainNode.__str__()

    def __repr__(self):
        return str(self)


class Node(BaseClassifier):
    def __init__(self, attributes, labels, classKey='class', F=-1):
        super(Node, self).__init__()
        self._attributes = attributes
        self._labels = labels
        self._classKey = classKey
        self.F = F

    def _reset(self):
        self.branches = {True: None, False: None}
        self.type = None # 'numerical' or 'categorical'
        self.nodeGini = None

    def fit(self, X):
        self.nodeGini = gini_index(X, classKey=self._classKey)
        gain = self.__find_best_split(X)
        if gain == 0:
            return

        true_split, false_split = self.__split(X)

        self.branches[True] = self.__create_branch(true_split)
        self.branches[False] = self.__create_branch(false_split)

        return self

    def __create_branch(self, X):
        if len(X) == 1:
            self.__create_leaf(X)
        node = Node(self._attributes, self._labels, classKey=self._classKey, F=self.F)
        ret = node.fit(X)
        if ret is None:
            return self.__create_leaf(X)
        else:
            return node

    def __create_leaf(self, X):
        leaf = Leaf(classKey=self._classKey)
        return leaf.fit(X)

    def __find_best_split(self, X):
        best_gain = 0
        best_feature, best_value = None, None
        if self.F < 0 or len(self._attributes) > self.F:
            features = self._attributes
        else:
            features = random.sample(self._attributes, k=self.F)
        for feature in features:
            for value in X[feature].unique():
                true_split, false_split = self.__try_split(X, feature, value)

                if len(true_split) == 0 or len(false_split) == 0:
                    continue

                # extract gain
                gain = delta_gini(true_split, false_split, self.nodeGini, classKey=self._classKey)

                # compare to best gain
                if gain > best_gain:
                    best_gain, best_feature, best_value = gain, feature, value

        self.type = 'numerical' if is_numeric else 'categorical'
        self.feature = best_feature
        self.value = best_value
        return best_gain

    def __split(self, X):
        return self.__try_split(X, feature=self.feature, value=self.value)

    def __try_split(self, X, feature, value):
        mask = self.__numeric_mask(X, feature, value) if is_numeric(value) else self.__categorical_mask(X, feature, value)
        return X[mask], X[~mask]

    @staticmethod
    def __numeric_mask(X, feature, value):
        return X[feature] >= value

    @staticmethod
    def __categorical_mask(X, feature, value):
        return X[feature] == value

    def predict(self, X):
        if self.nodeGini is None:
            raise ValueError('fit the node first by calling node.fit(X)')
        raise NotImplementedError

    def __str__(self, level=0, clearlvls=None, key=None):
        clearlvls = [] if clearlvls is None else clearlvls

        padding = ''.join(['│   ']*(level))

        for i in clearlvls:
            if i == 0:
                padding = '    ' + padding[4:]
            else:
                padding = padding[:i*4] + '    ' + padding[(i+1)*4:]

        if level == 0:
            msg = f'({self.comp_as_str})\n'
        else:
            if len(clearlvls) > 0:
                lvl_char = '└' if clearlvls[-1] == (level-1) else '├'
            else:
                lvl_char = '├'

            msg = padding[:-4] + lvl_char + f'{key}─ ' + f'({self.comp_as_str})\n'

        for idx, (key, branch) in enumerate(self.branches.items()):
            key = 't' if key else 'f'
            if idx == len(self.branches)-1:
                clearlvls.append(level)
            if isinstance(branch, Node):
                msg += branch.__str__(level=level+1, clearlvls=clearlvls, key=key)
            elif isinstance(branch, Leaf):
                if idx < len(self.branches)-1:
                    msg += padding + f'├{key}─ {branch}\n'
                else:
                    msg += padding + f'└{key}─ {branch}\n'
            if idx == len(self.branches)-1:
                clearlvls.remove(level)
        return msg

    @property
    def comp_as_str(self):
        if self.type == 'numerical':
            return f'{self.feature} >= {self.value}'
        else:
            return f'{self.feature} == {self.value}'


class Tree(Node):
    # just another name for the same class. Just because does not make
    # sense from a naming perspective
    pass


class Leaf(BaseClassifier):
    def __init__(self, classKey='class'):
        super(Leaf, self).__init__()
        self.classKey = classKey

    def _reset(self):
        self.predictions = None

    def fit(self, X):
        self.predictions = X[self.classKey].value_counts() / len(X)
        self.predictions = dict(zip(self.predictions.index, self.predictions))
        return self

    def predict(self, X):
        pass

    def __str__(self):
        return str(self.predictions)

    def __repr__(self):
        return self.__str__()