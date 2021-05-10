from base_forest import BaseForestClassifier
from tree_units import Tree
import random
import numpy as np

class RandomForestClassifier(BaseForestClassifier):
    def _fit_tree(self, tree_idx, X):
        self.__restart_seed(tree_idx)
        X = self.__get_bootstrapped_dataset(X)
        tree = Tree(self._attributes, self._labels, classKey=self.classKey, F=self.F)
        return tree.fit(X)

    @staticmethod
    def __get_bootstrapped_dataset(X):
        return X.sample(n=len(X), axis='rows', replace=True)

    def __restart_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
