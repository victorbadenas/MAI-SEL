import copy
import json
import random
import logging

from base_classifier import BaseClassifier
from utils.data import is_numeric
from utils.math import delta_gini, gini_index

class Node(BaseClassifier):
    def __init__(self, attributes, labels, classKey='class', F=-1):
        super(Node, self).__init__()
        self._attributes = attributes
        self._labels = labels
        self._classKey = classKey
        self.F = F
        self.class_name = self.__class__.__name__

    def _reset(self):
        self.branches = {True: None, False: None}
        self.type = None # 'numerical' or 'categorical'
        self.nodeGini = None
        self.trained = False
        self.feature = None
        self.value = None

    def fit(self, X):
        self.nodeGini = gini_index(X, classKey=self._classKey)
        gain = self.__find_best_split(X)
        if gain == 0:
            return

        true_split, false_split = self.__split(X)

        self.branches[True] = self.__create_branch(true_split)
        self.branches[False] = self.__create_branch(false_split)

        self.trained = True
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

        if self.F < 0 or len(self._attributes) < self.F:
            # use all attributes
            features = self._attributes
        else:
            # sample F attributes
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

        self.type = 'numerical' if is_numeric(best_value) else 'categorical'
        self.feature = best_feature
        self.value = best_value.item() if hasattr(best_value, 'item') else best_value
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
        if not self.trained:
            raise ValueError('fit the node first by calling node.fit(X)')
        if self.type == 'numerical':
            branch = self.__compare_numerical(X)
        else:
            branch = self.__compare_categorical(X)
        return self.branches[branch].predict(X)

    def __compare_categorical(self, x):
        return x[self.feature] == self.value

    def __compare_numerical(self, x):
        return x[self.feature] >= self.value

    def __str__(self, level=0, clearlvls=None, key=None):
        if not self.trained:
            return 'Node/Tree not trained'
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

    def to_dict(self):
        jsonData = copy.deepcopy(self.__dict__)
        for k, branch in jsonData['branches'].items():
            jsonData['branches'][k] = branch.to_dict()
        return jsonData

    def get_feature_importance(self):
        counts = dict(zip(self._attributes, [0]*len(self._attributes)))
        for branch in self.branches.values():
            if isinstance(branch, Leaf):
                continue
            else:
                feat_counts = branch.get_feature_importance()
            for k in counts:
                counts[k] += feat_counts[k]
        counts[self.feature] += 1
        return counts

    def load(self, data_dict):
        branches = data_dict.pop('branches')

        for k, v in data_dict.items():
            if k in self.__dict__:
                setattr(self, k, v)

        for k in self.branches:
            k_str = str(k).lower()
            if branches[k_str]['class_name'] == 'Node':
                self.branches[k] = Node(None, None).load(branches[k_str])
            elif branches[k_str]['class_name'] == 'Leaf':
                self.branches[k] = Leaf(None).load(branches[k_str])
            else:
                raise ValueError('incorrect dictionary format')
        return self

class Tree(Node):
    # just another name for the same class. Just because does not make
    # sense from a naming perspective
    pass


class Leaf(BaseClassifier):
    def __init__(self, classKey='class'):
        super(Leaf, self).__init__()
        self.classKey = classKey
        self.class_name = self.__class__.__name__

    def _reset(self):
        self.predictions = None

    def fit(self, X):
        self.predictions = X[self.classKey].value_counts() / len(X)
        self.predictions = dict(zip(self.predictions.index, self.predictions))
        return self

    def predict(self, X):
        return self.predictions

    def __str__(self):
        return str({k: round(v, 2) for k, v in self.predictions.items()})

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.__dict__

    def load(self, leaf_dict):
        for k, v in leaf_dict.items():
            if k in self.__dict__:
                setattr(self, k, v)
        return self
