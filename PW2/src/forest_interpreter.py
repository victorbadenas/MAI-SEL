import json
import logging
import numpy as np
import multiprocessing as mp
from tree_units import Tree
from utils.data import filterNone

class ForestInterpreter:
    def load_dict(self, d, **overrides):
        for k, v in d.items():
            if k == 'trees':
                continue
            setattr(self, k, v)
        self._load_trees(d['trees'])
        for k, v in overrides.items():
            setattr(self, k, v)

        if self.n_jobs < 1:
            self.n_jobs = None

    def _load_trees(self, trees_dict):
        self.trees = []
        for tree_dict in trees_dict:
            if not tree_dict:
                continue
            t = Tree(None, None).load(tree_dict)
            self.trees.append(t)

    def predict(self, X):
        if self.trees is None:
            raise ValueError('fit has not been called or forest has not been loaded')
        if self.n_jobs == 1:
            predictions = [None]*self.num_trees
            for i, t in enumerate(self.trees):
                predictions[i] = self._predict_tree(t, X)
        else:
            from functools import partial
            with mp.Pool(self.n_jobs) as p:
                predictions = list(p.map(partial(self._predict_tree, X=X), self.trees))

        predictions = filterNone(predictions) # remove predictions from non filtered trees
        predictions = self._aggregate_predictions(predictions)
        return np.array(predictions)

    def _aggregate_predictions(self, predictions):
        agg_predictions = [dict(zip(self._labels, [0]*len(self._labels))) for _ in range(len(predictions[0]))]
        for final_prediction_idx in range(len(predictions[0])):
            tree_preds = [predictions[n_tree][final_prediction_idx] for n_tree in range(len(predictions))]
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


def forest_from_json(fp=None, d=None, **overrides):
    if fp is None and d is None:
        raise ValueError('Specify either a file path or a dictionary')
    elif fp is not None and d is not None:
        raise ValueError('Specify either a file path or a dictionary, not both')
    if fp is not None:
        with open(fp, 'r') as f:
            d = json.load(f)
    fi = ForestInterpreter()
    fi.load_dict(d, **overrides)
    return fi