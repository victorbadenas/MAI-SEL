from base_forest import BaseForestClassifier
from tree_units import Tree

class RandomForestClassifier(BaseForestClassifier):
    def _fit_tree(self, tree_idx, X):
        X = self.__get_bootstrapped_dataset(X)
        tree = Tree(self._attributes, self._labels, classKey=self.classKey, F=self.F)
        return tree.fit(X)

    @staticmethod
    def __get_bootstrapped_dataset(X):
        return X.sample(n=len(X), axis='rows', replace=True)
