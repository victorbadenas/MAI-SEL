from pathlib import Path

class Interpreter:
    def __init__(self, rules_path):
        self.rules_path = Path(rules_path)
        self.validate_arguments()

    def validate_arguments(self):
        assert self.rules_path.exist()

    def interpret_rule(rule:str):
        return