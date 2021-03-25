from pathlib import Path
from rule import Rule
import re
import pandas as pd

class RuleInterpreter:
    def __init__(self, rules_path):
        self.rules_path = Path(rules_path)
        self.validate_arguments()
        self.rules = []
        self.load_rules()

    def predict(self, X):
        data = self.__validate_inference_data(X)
        return self._predict(data)

    def _predict(self, data):
        predicted_labels = len(data) * [None]
        for idx, item in enumerate(data):
            for rule in self.rules:
                if rule.is_covered(item):
                    predicted_labels[idx] = rule.label
                    break
        return predicted_labels

    def load_rules(self):
        with open(self.rules_path, 'r') as f:
            for line in f:
                if line.startswith("IF"):
                    self.rules.append(self.interpret_rule(line))

    def validate_arguments(self):
        assert self.rules_path.exists()

    @staticmethod
    def interpret_rule(rule_str:str):
        rule_str = rule_str.strip()
        if '\t' in rule_str:
            rule_str = rule_str.split('\t')
            accuracy, coverage = eval(rule_str[-1])
            rule_str = rule_str[0]

        if ' AND ' in rule_str and ' OR ' in rule_str:
            raise ValueError('rule with and and or not implemented')
        elif ' OR ' in rule_str:
            rule = Rule(type_='OR')
        else:
            rule = Rule(type_='AND')

        rule.p, rule.t = accuracy*coverage, coverage

        rule_list = rule_str.split()
        idx = 0
        while idx < len(rule_list):
            if rule_list[idx].lower() in ("if", "and", "or"):
                if idx+3 < len(rule_list):
                    attribute = rule_list[idx+1]
                    value = rule_list[idx+3]
                    rule[attribute] = value
                idx += 3
            elif rule_list[idx].lower() == "then":
                rule._target_attribute = rule_list[idx+1]
                rule.class_ = rule_list[idx+3]
                break
            idx += 1
        return rule

    def __validate_inference_data(self, X):
        if isinstance(X, pd.DataFrame):
            X.columns = X.columns.astype(str)
            return X.to_dict('records')
        else:
            raise NotImplementedError
