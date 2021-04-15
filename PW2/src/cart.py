import pandas as pd
from utils.math import gini_index
from base_classifier import BaseClassifier

class CART(BaseClassifier):
    def __init__(self):
        super(CART, self).__init__()

    def _reset(self):
        pass

    def load(self, path_to_file):
        pass

    def _fit(self, data):
        dataset_gini = gini_index(data)
        logging.debug(f'Full dataset {}')

    def _predict(self, X):
        pass

    def _validate_train_data(self, X, Y):
        if isinstance(X, pd.DataFrame):
            if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
                data = X.join(Y)
                return data.to_dict('records')
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _save_to_json(self, path_to_file, data):
        pass

    def _save_to_txt(self, path_to_file):
        pass
