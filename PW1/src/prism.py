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
        self.__computeAvailableAttributes()
        self.class_ = class_
        self.precision = 0.0
        self.coverage = 0.0

    def __computeAvailableAttributes(self):
        self.availableAttributes = np.where(self.antecedent == -1)[0]

    def isPerfect(self):
        return self.precision == 1.0

    def areAvailableAttributes(self):
        return len(self.availableAttributes) != 0

    def getAvailableAttributes(self):
        return self.availableAttributes

    def getCoveredInstances(self, X):
        m1 = X == self.antecedent  # check attributes that match
        m2 = self.antecedent == -1  # attributes excluded
        return np.all(m1 + m2, axis=1)  # or and check which instances are covered by the rule

    def evaluate(self, X, Y):
        coveredInstances = self.getCoveredInstances(X)
        self.coverage = np.sum(coveredInstances)/X.shape[0]
        self.accuracy = np.sum((coveredInstances) * (Y == self.class_))/X.shape[0]

    def __setitem__(self, index, value):
        self.antecedent[index] = value
        self.__computeAvailableAttributes()

    def __gettem__(self, index):
        return self.antecedent[index]

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        if len(self.antecedent) == len(self.availableAttributes):
            return "Empty Rule"
        usedAttributes = np.where(self.antecedent != -1)[0]
        string_ = "Rule: IF"
        for attr in usedAttributes:
            string_ += f" {int(attr)} IS {int(self.antecedent[attr])} AND"
        else:
            string_ = string_[:-3] + f"THEN {int(self.class_)}"
        return string_

    def __repr__(self):
        return str(self)

    def __len__(self):
        return np.sum(self.antecedent != -1)


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
        allPossibleValues = list(map(lambda idx: np.unique(X[:,idx]), range(X.shape[1])))
        self.rules = list()
        for class_ in np.unique(Y):
            logging.debug(f"Finding rules for class: {class_}")
            E = X.copy()
            while E.size != 0:
                logging.debug(f"E.shape {E.shape}")

                # build rule
                R = Rule(X.shape[1], class_)
                while not R.isPerfect() and R.areAvailableAttributes():
                    # Rop = R.copy()
                    # bestPrecision = 0.0
                    allRules = list()
                    for attribute in R.getAvailableAttributes():
                        for value in allPossibleValues[attribute]:
                            Rav = Rule(X.shape[-1], class_)
                            Rav[attribute] = value
                            Rav.evaluate(X, Y)
                            allRules.append((attribute, value, Rav))

                    # Rop = self._getBestRule()
                    Rop = allRules[0]
                    for Rav in allRules[1:]:
                        att, val, rule = Rav
                        if Rop[2].accuracy < rule.accuracy:
                            Rop = Rav
                        elif Rop[2].accuracy == rule.accuracy:
                            if Rop[2].coverage < rule.coverage:
                                Rop = Rav

                    R[Rop[0]] = Rop[1]
                logging.debug(R)
                E = self.__removeCoveredInstances(E, R)
                logging.info(f"Added Rule: {R}")
                self.rules.append(R)
        return self

    def __removeCoveredInstances(self, E, R):
        return E[~R.getCoveredInstances(E)]

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
