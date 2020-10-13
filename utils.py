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


def transpose_matrix(x):
    return np.transpose(x)


def conditioning_angle(a_, b_, x_):
    """
    if this value is little, we know that the angle between b and Ax is little and so b is close to the ImA, and
    the problem is well conditioned
    """
    return np.arccos(np.divide(norm(np.matmul(a_, x_)), norm(b_)))
