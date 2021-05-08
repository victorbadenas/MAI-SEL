from base_forest import BaseForestClassifier
from tree_units import Tree

class DecisionForestClassifier(BaseForestClassifier):
    def _fit_tree(self, tree_idx, X):
        X = self.__select_features_from_dataset(X)
        attributes = list(X.columns)[:-1]
        tree = Tree(attributes, self._labels, classKey=self.classKey, F=-1)
        return tree.fit(X)

    def __select_features_from_dataset(self, X):
        X, Y = X.drop(self.classKey, axis=1), X[self.classKey]
        n_features = len(X.columns)
        F = self.F(n_features) if callable(self.F) else self.F
        F = min(F, n_features)
        return X.sample(n=F, axis='columns').join(Y)

    def _save_to_json(self, path_to_file):
        if callable(self.F):
            self.F = 'callable'
        super()._save_to_json(path_to_file)
