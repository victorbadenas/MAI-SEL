from .base_classifier import BaseClassifier

class DecisionForestClassifier(BaseClassifier):
    def __init__(self, headers=None):
        super(self, DecisionForestClassifier).__init__(self, headers)
