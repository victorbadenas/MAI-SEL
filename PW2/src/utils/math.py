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
        counts, gini_indexes[idx] = class_count_gini(filtered_df, classKey)
        logging.debug(f'gini_index({feature})={gini_indexes[idx]}. Counts:\n {counts}\n')
    return gini_indexes

def class_count_gini(df, classKey='class'):
    assert len(df.columns) == 2, f"columns should have len 2: [attribute, classKey]"
    assert classKey in df.columns, f'class atribute not in dataframe'
    attribute = list(filter(lambda x: x != classKey, df.columns))[0]

    counts = df.pivot_table(index=attribute, columns=classKey, aggfunc='size', fill_value=0.0)
    sum_ = np.sum(counts.to_numpy(), axis=1, keepdims=True)
    gini_table = 1 - ((counts / sum_)**2).sum(axis=1)
    gini_value = np.sum(gini_table * sum_[:,0]) / np.sum(sum_)
    return gini_table, gini_value
