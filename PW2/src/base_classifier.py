import json
import logging
from pathlib import Path


class BaseClassifier:
    """Abstract base classifier
    """
    def __init__(self, headers=None):
        self._reset()
        self._target_attribute = 'target'
        self._labels = headers
        self._attributes = None

    """
    public methods
    """
    def fit(self, X, Y):
        data = self._validate_train_data(X, Y)
        return self._fit(data)

    def predict(self, X):
        self._predict(X)

    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict(X)

    def save(self, path_to_file):
        path_to_file = Path(path_to_file)
        path_to_file.parent.mkdir(parents=True, exist_ok=True)
        if path_to_file.suffix == '.json':
            self._save_to_json(path_to_file)
        else:
            self._save_to_txt(path_to_file)

    def load(self, path_to_file):
        raise NotImplementedError

    """
    abstract methods
    """
    def _save_to_json(self, path_to_file, data):
        raise NotImplementedError

    def _save_to_txt(self, path_to_file):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _fit(self, data):
        raise NotImplementedError

    def _predict(self, X):
        raise NotImplementedError

    def _validate_train_data(self, X, Y):
        raise NotImplementedError

    """
    properties
    """
    @property
    def attributes(self):
        return self._attributes

    @property
    def labels(self):
        return self._labels
