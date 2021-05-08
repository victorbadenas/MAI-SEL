import copy
import json
import logging
import multiprocessing as mp
import random
from pprint import pformat, pprint

import numpy as np
import pandas as pd

from base_classifier import BaseClassifier
from utils.data import is_numeric
from utils.math import delta_gini, gini_index


class RandomForestClassifier(BaseClassifier):
    def __init__(self, headers=None, F=-1, num_trees=10, classKey=None, n_jobs=1):
        super(RandomForestClassifier, self).__init__(headers)
        self.type = self.__class__.__name__
        self.classKey = classKey
        self.F = F
        self.num_trees = num_trees
        self.n_jobs = None if n_jobs < 1 else n_jobs

    def _reset(self):
        self.trees = None

    def load(self, path_to_file):
        pass

    def _fit(self, X):
        self._attributes = list(X.columns)
        self._attributes.remove(self.classKey)
        self._labels = list(X[self.classKey].unique())

        # tree init
        self.trees = [Tree(
                    self._attributes, self._labels, classKey=self.classKey, F=self.F
                ) for _ in range(self.num_trees)]

        if self.n_jobs == 1:
            for t in self.trees:
                t = self._fit_tree(t, X)
        else:
            from functools import partial
            with mp.Pool(self.n_jobs) as p:
                self.trees = p.map(partial(self._fit_tree, X=X), self.trees)
        if not all(t.trained for t in self.trees):
            fitted = sum(t.trained for t in self.trees)
            logging.warning(f'Not all the trees have been fitted. {len(self.trees) - fitted} of {len(self.trees)} have not been fitted')

    def _fit_tree(self, tree, X):
        X = self.__get_bootstrapped_dataset(X)
        return tree.fit(X)

    @staticmethod
    def __get_bootstrapped_dataset(X):
        return X.sample(n=len(X), axis='rows', replace=True)

    def _predict(self, X):
        if self.trees is None:
            raise ValueError('fit has not been called')
        if self.n_jobs == 1:
            predictions = [None]*self.num_trees
            for i, t in enumerate(self.trees):
                predictions[i] = self._predict_tree(t, X)
        else:
            from functools import partial
            with mp.Pool(self.n_jobs) as p:
                predictions = list(p.map(partial(self._predict_tree, X=X), self.trees))
        predictions = list(filter(lambda i: i is not None, predictions)) # remove predictions from non filtered trees
        predictions = self._aggregate_predictions(predictions)
        return np.array(predictions)

    def _aggregate_predictions(self, predictions):
        agg_predictions = [dict(zip(self._labels, [0]*len(self._labels))) for _ in range(len(predictions[0]))]
        for final_prediction_idx in range(len(predictions[0])):
            tree_preds = [predictions[n_tree][final_prediction_idx] for n_tree in range(self.num_trees)]
            for pred in tree_preds:
                for label in self._labels:
                    agg_predictions[final_prediction_idx][label] += pred.get(label, 0)
            agg_predictions[final_prediction_idx] = {k: v/len(predictions) for k, v in agg_predictions[final_prediction_idx].items()}
        final_predictions = [max(pred, key=pred.get) for pred in agg_predictions]
        return final_predictions

    def _predict_tree(self, tree, X):
        if not tree.trained:
            return 
        return [tree.predict(x) for x in X.to_dict('records')]  # convert to list(dict()) for speeding up iterations

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

    def get_feature_importance(self):
        if self.trees is None:
            logging.error('Model not fitted')
            raise ValueError('Model not fitted')
        counts = dict(zip(self._attributes, [0]*len(self._attributes)))
        for tree in self.trees:
            if not tree.trained:
                continue
            tree_counts = tree.get_feature_importance()
            for k in counts:
                counts[k] += tree_counts[k]
        return counts


class Node(BaseClassifier):
    def __init__(self, attributes, labels, classKey='class', F=-1):
        super(Node, self).__init__()
        self._attributes = attributes
        self._labels = labels
        self._classKey = classKey
        self.F = F
        self.type = self.__class__.__name__

    def _reset(self):
        self.branches = {True: None, False: None}
        self.type = None # 'numerical' or 'categorical'
        self.nodeGini = None
        self.trained = False

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
            jsonData['branches'] = branch.to_dict()
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

class Tree(Node):
    # just another name for the same class. Just because does not make
    # sense from a naming perspective
    pass


class Leaf(BaseClassifier):
    def __init__(self, classKey='class'):
        super(Leaf, self).__init__()
        self.classKey = classKey
        self.type = self.__class__.__name__
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
        return self.predictions
