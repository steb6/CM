import numpy as np


def norm(x):
    return np.sqrt(np.sum(np.square(x)))


def row_to_column(x):
    x.shape = (x.shape[0], 1)
    return x


def column_to_row(x):
    y = np.copy(x)
    y.shape = (1, y.shape[0])
    return y
