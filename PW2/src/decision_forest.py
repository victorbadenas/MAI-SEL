from base_forest import BaseForestClassifier
from tree_units import Tree
import random
import numpy as np

class DecisionForestClassifier(BaseForestClassifier):
    def _fit_tree(self, tree_idx, X):
        self.__restart_seed(tree_idx)
        X = self.__select_features_from_dataset(tree_idx, X)
        attributes = list(X.columns)[:-1]
        tree = Tree(attributes, self._labels, classKey=self.classKey, F=-1)
        return tree.fit(X)

    def __select_features_from_dataset(self, id, X):
        X, Y = X.drop(self.classKey, axis=1), X[self.classKey]
        n_features = len(X.columns)
        if self.F == 'random':
            F = random.randint(1, n_features+1)
        else:
            F = self.F
        F = min(F, n_features)
        return X.sample(n=F, axis='columns').join(Y)

    def __restart_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _save_to_json(self, path_to_file):
        super()._save_to_json(path_to_file)
