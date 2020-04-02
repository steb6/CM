import numpy as np
from utils import transpose_matrix


def conjugate_gradient(A, b):
    x = 0
    r = d = b
    for k in range(len(A)):
        r_T = transpose_matrix(r)
        d_T = transpose_matrix(d)
        above = np.matmul(r_T, r)
        under = np.matmul(d_T, np.matmul(A, d))
        alpha = np.divide(above, under)
        x = x + alpha*np.matmul(A, d)
        beta = np.divide(np.matmul(np.transpose(r), r), np.matmul(np.transpose(r), r))
        d = r + beta*d
    return x
