import numpy as np
from numpy import genfromtxt


def read_data(filename):
    # Read data
    A = genfromtxt(filename, delimiter=',')

    # Print dimensions
    #print('Readed {} rows and {} column'.format(A.shape[0], A.shape[1]))
    #print('First line: '+str(A[0]))

    # Remove indexes
    A = np.delete(A, 0, 1)

    # Print new dimensions
    #print('Deleted indexes, new dimensions are {} rows and {} column'.format(A.shape[0], A.shape[1]))
    #print('First line: '+str(A[0]))

    # Add derived columns
    A_1 = np.sum(A, 1)
    A = np.c_[A, A_1]
    #print('Added new columns, new dimensions are {} rows and {} column'.format(A.shape[0], A.shape[1]))
    #print('First line: '+str(A[0]))
    return A
