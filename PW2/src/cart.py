import logging
import pandas as pd
import numpy as np
import json
import copy
from pprint import pformat, pprint
from utils.math import gini_index
from base_classifier import BaseClassifier

class Leaf:
    def __init__(self, attributes, labels, classKey='class'):
        self._attributes = attributes
        self._labels = labels
        self._classKey = classKey

    def fit(self, data):
        self.fit_attribute(data)
        self.fit_value(data)
        self.create_leaves(data)

    def fit_attribute(self, data):
        gini_values = gini_index(data)
        min_idx = np.argmin(gini_values)
        self._attribute = self.attributes[min_idx]

    def fit_value(self, data):
        filtered_df = data[[self._attribute, 'class']]
        counts = filtered_df.pivot_table(index=self._attribute, columns='class', aggfunc='size', fill_value=0.0)
        sum_ = np.sum(counts.to_numpy(), axis=1, keepdims=True)
        gini = 1 - ((counts / sum_)**2).sum(axis=1)
        idx = np.argmin(gini)
        self._value = counts.index[idx]

    def create_leaves(self, data):
        pos = data[data[self._attribute] == self._value]
        neg = data[data[self._attribute] != self._value]
        possible_values = [self._value, 'other']
        self.branches = dict(zip(possible_values, [None]*2))
        for branch, branchdata in zip(possible_values, [pos, neg]):
            classes = branchdata[self._classKey]
            if (classes == classes.iloc[0]).all():
                # assign class string label as end of the branch.
                self.branches[branch] = classes.iloc[0]
            elif (branchdata.drop(self._classKey, axis=1) == branchdata.drop(self._classKey, axis=1).iloc[0]).all().all():
                counts = classes.value_counts()
                self.branches[branch] = classes.value_counts().index[0]
                logging.warning(f'reached final node with conflicting data, assigning {self.branches[branch]} with {100*counts[0]/counts.sum():.4f}% confidence')
            else:
                self.__create_new_leaf(branch, branchdata)

    def __create_new_leaf(self, branch, branchdata):
        attributes = self.__get_attributes_from_dataset(branchdata)
        labels = self.__get_labels_from_dataset(branchdata)
        self.branches[branch] = Leaf(attributes, labels)
        self.branches[branch].fit(branchdata)

    def __get_labels_from_dataset(self, data):
        return list(data[self._classKey].unique())

    def __get_attributes_from_dataset(self, data):
        attributes = list(data.columns)
        attributes.remove(self._classKey)
        return attributes

    @property
    def labels(self):
        return self._labels

    @property
    def attributes(self):
        return self._attributes

    def to_dict(self):
        data = copy.deepcopy(self.__dict__)
        for branch, value in data['branches'].items():
            if isinstance(value, Leaf):
                data['branches'][branch] = value.to_dict()
        return data

    def __str__(self, level=0, clearlvls=None):
        clearlvls = [] if clearlvls is None else clearlvls

        padding = ''.join(['│   ']*(level))

        for i in clearlvls:
            if i == 0:
                padding = '    ' + padding[4:]
            else:
                padding = padding[:i*4] + '    ' + padding[(i+1)*4:]

        if level == 0:
            msg = f'({self._attribute}, branches: {list(self.branches)})\n'
        else:
            if len(clearlvls) > 0:
                lvl_char = '└' if clearlvls[-1] == (level-1) else '├'
            else:
                lvl_char = '├'

            msg = padding[:-4] + lvl_char + '── ' + f'({self._attribute}, branches: {list(self.branches)})\n'

        for idx, branch in enumerate(self.branches.values()):
            if idx == len(self.branches)-1:
                clearlvls.append(level)
            if isinstance(branch, Leaf):
                msg += branch.__str__(level=level+1, clearlvls=clearlvls)
            elif idx < len(self.branches)-1:
                msg += padding + f'├── {branch}\n'
            else:
                msg += padding + f'└── {branch}\n'
            if idx == len(self.branches)-1:
                clearlvls.remove(level)
        return msg

    def __repr__(self):
        return str(self)


class CART(BaseClassifier):
    def __init__(self, classKey='class'):
        super(CART, self).__init__()
        self.classKey = classKey
        self.mainLeaf = None

    def _reset(self):
        pass

    def load(self, path_to_file):
        pass

    def _fit(self, data):
        self._attributes = list(data.columns)
        self._attributes.remove(self.classKey)
        self._labels = list(data[self.classKey].unique())
        self.mainLeaf = Leaf(self._attributes, self._labels, classKey=self.classKey)
        self.mainLeaf.fit(data)


    def _predict(self, X):
        pass

    def _validate_train_data(self, X, Y=None):
        if isinstance(X, pd.DataFrame):
            if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
                assert X.shape[0] == Y.shape[0], 'instance numbers inconsistent'
                return X.join(Y)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _save_to_json(self, path_to_file):
        jsondata = copy.deepcopy(self.__dict__)
        jsondata['mainLeaf'] = jsondata['mainLeaf'].to_dict()
        with open(path_to_file, 'w') as f:
            json.dump(jsondata, f, indent=4)

    def _save_to_txt(self, path_to_file):
        msg = str(self)
        with open(path_to_file, 'w') as f:
            f.write(msg)

    def __str__(self):
        return self.mainLeaf.__str__()

    def __repr__(self):
        return str(self)
