from collections import Counter

def gini_index(data, classKey='class'):
    class_counts = list(Counter([item[classKey] for item in data]).values())
    return 1 - sum((item/len(data))**2 for item in class_counts)

