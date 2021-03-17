import os
import sys
import copy
import warnings
import logging
import numpy as np
import pandas as pd

class Rule:
    def __init__(self, shape, class_):
        self.antecedent = np.full(shape, -1, dtype=np.int)
        self.availableAttributes = np.where(self.antecedent == -1)[0]
        self.class_ = class_
        self.precision = 0.0

    def isPerfect(self):
        return self.precision != 1.0

    def areAvailableAttributes(self):
        return len(self.availableAttributes) != 0

    def getAvailableAttributes(self):
        return self.availableAttributes

    def evaluate(self, data):
        pass

    def __setattr__(self, index, value):
        self.antecedent[index] = value

    def copy(self):
        return copy.deepcopy(self)

class Prism:
    def __init__(self):
        pass

    def fit(self, X, Y):
        X, Y = self.__validate_data(X, Y)
        logging.debug(f"X Type: ({type(X)},{X.dtype}), X shape:{X.shape}")
        logging.debug(f"Y Type: ({type(Y)},{Y.dtype}), Y shape:{Y.shape}")
        return self._fit(X, Y)

    def predict(self, X):
        pass

    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict(X)

    def _fit(self, X, Y):
        self.allPossibleValues = list(map(lambda idx: np.unique(X[:,idx]), range(X.shape[1])))
        self.rules = list()
        for class_ in np.unique(Y):
            E = X[Y==class_]
            while E.size != 0:
                R = Rule(X.shape[1], class_)
                while not R.isPerfect(None) and R.areAvailableAttributes():
                    Rop = R.copy()
                    for attribute in R.getAvailableAttributes():
                        possibleValues = self.possibleValues[attribute]
                        for value in possibleValues:
                            Rav = R.copy()
                            Rav[attribute] = value
                            R.evaluate()
                            if Rav.precision > Rop.precision:
                                Rop = Rav
                    R = Rop
                E = self.__removeCoveredInstances(E, R)
                self.rules.append(R)
        return self


    def __validate_data(self, X, Y):
        X = self.__safe_conversion(X)
        Y = self.__safe_conversion(Y)
        if X.ndim != 2:
            raise ValueError('X must be a 2 dimension array')
        if Y.ndim != 1:
            raise ValueError('Y must be a 1d array')
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f'inconsistent shapes of X:{X.shape} and Y:{Y.shape}')
        return X, Y

    def __safe_conversion(self, array):
        if isinstance(array, pd.DataFrame):
            array = array.to_numpy()
        elif isinstance(array, pd.Series):
            array = np.array(array)
        elif isinstance(array, list):
            array = np.array(array)
        elif isinstance(array, np.ndarray):
            return array
        else:
            TypeError('array should be either a pd.DataFrame, np.ndarray or list of lists')

        if array.dtype != np.int:
            msg = "Input array is not int, will be casted to int"
            logging.warning(msg)
            warnings.warn(msg)
            array = array.astype(np.int)

        return array
