def is_numeric(x):
    return isinstance(x, int) or isinstance(x, float)

def filterNone(x):
    return list(filter(lambda i: i is not None, x))