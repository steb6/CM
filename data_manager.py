import numpy as np
from numpy import genfromtxt


def read_data(filename):
    # Read data
    A = genfromtxt(filename, delimiter=',')

    # Print dimensions
    print('Readed {} rows and {} column'.format(A.shape[0], A.shape[1]))
    #print('First line: '+str(A[0]))

    # Remove indexes
    A = np.delete(A, 0, 1)
    y = A.shape[1]
    b = A[:, y-1]
    A = np.delete(A, np.s_[y-2, y], axis=1)

    # Print new dimensions
    #print('Deleted indexes, new dimensions are {} rows and {} column'.format(A.shape[0], A.shape[1]))
    #print('First line: '+str(A[0]))

    # Add derived columns
    A_1 = np.log(np.abs(A[:, 0]))
    A_2 = A[:, 1] * A[:, 2] * A[:, 3]
    A_3 = np.square(A[:, 4])
    A = np.c_[A, A_1, A_2, A_3]
    #print('Added new columns, new dimensions are {} rows and {} column'.format(A.shape[0], A.shape[1]))
    #print('First line: '+str(A[0]))
    return A, b
