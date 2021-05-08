from base_forest import BaseForestClassifier
from tree_units import Tree

class RandomForestClassifier(BaseForestClassifier):
    def _fit_tree(self, tree, X):
        X = self.__get_bootstrapped_dataset(X)
        return tree.fit(X)

    def _init_trees(self):
        self.trees = [Tree(
            self._attributes, self._labels, classKey=self.classKey, F=self.F
        ) for _ in range(self.num_trees)]

    @staticmethod
    def __get_bootstrapped_dataset(X):
        return X.sample(n=len(X), axis='rows', replace=True)
