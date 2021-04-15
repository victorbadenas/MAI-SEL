import logging
from collections import Counter
import numpy as np

def gini_index(data, classKey='class'):
    columns = list(data.columns)
    if classKey in columns:
        columns.remove(classKey)
    gini_indexes = [None]*len(columns)
    for idx, feature in enumerate(columns):
        filtered_df = data[[feature, 'class']]
        counts = filtered_df.pivot_table(index=feature, columns='class', aggfunc='size', fill_value=0.0)
        sum_ = np.sum(counts.to_numpy(), axis=1, keepdims=True)
        class_gini = 1 - ((counts / sum_)**2).sum(axis=1)
        gini_indexes[idx] = np.sum(class_gini * sum_[:,0]) / np.sum(sum_)
        logging.debug(f'gini_index({feature})={gini_indexes[idx]}. Counts:\n {counts}\n')
    return gini_indexes

