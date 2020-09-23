import numpy as np
from numpy import genfromtxt


def read_data(filename, add_augmented_columns=True):
    # Read data
    A = genfromtxt(filename, delimiter=',')

    # Print dimensions
    # print('Readed {} rows and {} column'.format(A.shape[0], A.shape[1]))

    # Remove indexes
    A = A[:, 1:]
    # Extract last 2 columns for b
    b = A[:, -2:]
    # But b must have a single dimension
    b = b[:, 0]
    # Remove last 2 columns from A
    A = A[:, :-2]

    # Add derived columns
    if add_augmented_columns:
        A_1 = np.log(np.abs(A[:, 0]))
        A_2 = A[:, 1] * A[:, 2] * A[:, 3]
        A_3 = np.square(A[:, 4])
        A = np.c_[A, A_1, A_2, A_3]

    return A, b
