import numpy as np
from utils import transpose_matrix


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0) and is_symmetric(x)


def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def conjugate_gradient(A, b):
    if is_pos_def(np.matmul(transpose_matrix(A), A)):
        print("A_T*A is positive definite")
    else:
        print("A_T*A is not positive definite")
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
