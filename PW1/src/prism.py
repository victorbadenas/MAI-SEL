import os
import sys
import copy
import warnings
import logging
import pandas as pd
from itertools import filterfalse
from rule import Rule


class Prism:
    def __init__(self):
        self.reset()
        self._target_attribute = 'target'
        self._labels = []

    def fit(self, X, Y):
        data = self.__validate_data(X, Y)
        return self._fit(data)

    def predict(self, X):
        pass

    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict(X)

    def reset(self):
        self._rules = []

    def _fit(self, data):
        self.reset()
        for class_ in self.labels:
            E = data[:]
            while self.__class_in_instances(E, class_):
                rule = self.__build_rule(class_, E)
                self.rules.append(rule)
                E = self.__remove_covered_instances(E, rule)
        return self

    def __build_rule(self, class_, instances):
        rule = Rule(class_, self._attributes, self._target_attribute, initial_coverage=len(instances))
        E = instances[:]
        while not rule.is_perfect() and rule.coverage > 1:
            allRules = []
            for attribute in rule.unused_attributes:
                for value in self.__get_possible_values(E, attribute):
                    rav = Rule(class_, self._attributes, self._target_attribute)
                    rav[attribute] = value
                    rav.evaluate(E)
                    allRules.append(rav)
            bestRule = max(allRules)
            rule.extend(bestRule)
            E = rule.apply(E)
            rule.evaluate(E)
        return rule

    def __class_in_instances(self, instances, class_):
        # loop over instances and see if any matches the class checked
        return any(map(lambda x: x[self._target_attribute] == class_, instances))

    def __get_possible_values(self, instances, attr):
        # returns possible values for a given attributes
        values = set()
        for instance in instances:
            if instance[attr] not in values:
                values.add(instance[attr])
        return values

    # This method remove all instances covered by the set of rules
    def __remove_covered_instances(self, instances, rule):
        return list(filterfalse(rule.is_covered, instances))

    def __validate_data(self, X, Y):
        self.__extract_attributes(X)
        self._labels = self.__extract_labels(Y)
        data = self.__format_data_to_dict(X, Y)
        return data

    def __extract_labels(self, Y):
        if isinstance(Y, pd.Series):
            return list(Y.unique())
        elif isinstance(Y, pd.DataFrame):
            return self.__extract_labels(Y[Y.columns[0]])
        raise NotImplementedError

    def __extract_target_attribute(self, Y):
        if isinstance(Y, pd.Series):
            self._target_attribute = Y.name
        elif isinstance(Y, pd.DataFrame):
            self._target_attribute = Y.columns[0]
        else:
            raise NotImplementedError

    def __extract_attributes(self, X):
        if isinstance(X, pd.DataFrame):
            self._attributes = list(X.columns.astype(str))
        else:
            raise NotImplementedError

    def __format_data_to_dict(self, X, Y):
        if isinstance(X, pd.DataFrame):
            if isinstance(Y, pd.DataFrame):
                if Y.shape[-1] == 1:
                    return self.__format_data_to_dict(X, Y[Y.columns[0]])
                else:
                    raise ValueError('Wrong dimensions for Y. {Y.shape} is not (n_instances, 1)')
            elif isinstance(Y, pd.Series):
                X = pd.concat((X, Y), axis=1)
                X.columns = X.columns.astype(str)
                return X.to_dict('records')
            raise NotImplementedError('currently only supporting pd.DataFrame or pd.series for taget variable')
        raise TypeError('array should be either a pd.DataFrame')

    @property
    def rules(self):
        return self._rules

    @property
    def attributes(self):
        return self._attributes

    @property
    def labels(self):
        return self._labels
