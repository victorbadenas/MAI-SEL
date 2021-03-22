class Rule:
    def __init__(self, class_, available_attibutes, target_attribute, initial_coverage=0.0):
        self._available_attibutes = available_attibutes
        self._target_attribute = target_attribute
        self.antecedent = {}
        self.class_ = class_
        self.p = 0.0
        self.t = initial_coverage

    @property
    def stats(self):
        return self.p, self.t

    @property
    def accuracy(self):
        return -1.0 if self.t <= 0 else self.p / self.t

    @property
    def coverage(self):
        return self.t

    @property
    def used_attributes(self):
        return list(self.antecedent.keys())

    @property
    def unused_attributes(self):
        return sorted(set(self._available_attibutes) - set(self.used_attributes))

    def is_perfect(self):
        return self.accuracy == 1.0

    def items(self):
        return self.antecedent.items()

    def keys(self):
        return self.antecedent.keys()

    def extend(self, other):
        assert isinstance(other, Rule)
        for k, v in other.items():
            if k not in self.antecedent:
                self[k] = v

    def evaluate(self, data):
        covered = self.__compute_covered(data)
        correct = self.__compute_correct(covered)
        self.p, self.t = len(correct), len(covered)

    def apply(self, data):
        return self.__compute_covered(data)

    def is_covered(self, instance):
        return self.__instance_match(instance)

    def __compute_covered(self, instances):
        # iterate over instances and filter the matched ones
        return list(filter(self.__instance_match, instances))

    def __instance_match(self, instance):
        # check if all requirements in the antecedent are met
        return all([instance[k] == v for k, v in self.antecedent.items()])

    def __compute_correct(self, instances):
        # iterate over instances and filter the ones where the class matches
        return list(filter(self.__class_match, instances))

    def __class_match(self, instance):
        # check if the class matches
        return instance[self._target_attribute] == self.class_

    def __len__(self):
        return len(self.antecedent)

    def __setitem__(self, key, value):
        self.antecedent[key] = value

    def __getitem__(self, key):
        return self.antecedent[key]

    def __eq__(self, other):
        return self.accuracy == other.accuracy and self.coverage == other.coverage

    def __lt__(self, other):
        coverage_compare = self.accuracy == other.accuracy and self.coverage < other.coverage
        return self.accuracy < other.accuracy or coverage_compare

    def __gt__(self, other):
        coverage_compare = self.accuracy == other.accuracy and self.coverage > other.coverage
        return self.accuracy > other.accuracy or coverage_compare

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __str__(self):
        self.sort()
        formatted_rule = "IF"
        for k, v in self.items():
            formatted_rule += f" {k} IS {v} AND"
        else:
            formatted_rule = formatted_rule[:-4] + f" THEN {self._target_attribute} IS {self.class_}"
        return formatted_rule

    def __repr__(self):
        return str(self)

    def sort(self):
        sorted_keys = sorted(self.keys())
        sorted_antecedent = dict()
        for k in sorted_keys:
            sorted_antecedent[k] = self[k]
        self.antecedent = sorted_antecedent

    def from_dict(self, dict_):
        for k, v in dict_.items():
            if k in self.__dict__:
                setattr(self, k, v)
