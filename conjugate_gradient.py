import numpy as np
from utils import transpose_matrix, norm


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

# algorithm from https://www.stat.washington.edu/wxs/Stat538-w03/conjugate-gradients.pdf
def conjugate_gradient1(A, b):
    if is_pos_def(np.matmul(transpose_matrix(A), A)):
        print("A_T*A is positive definite")
    else:
        print("A_T*A is not positive definite")
    x = np.zeros(A.shape[1])
    r =  b  - np.matmul(A, x)
    g_0 = np.matmul(transpose_matrix(A), r)
    for i in range(1, A.shape[1]):
        if i>1:
            g_2 = g_1
            g_1 = g
        else:
            g = g_1 = g_0
        if not np.any(g):
            return x
        if i>1:
            beta = -np.divide(np.square(norm(g_1)),np.square(norm(g_2)))
        if i==1: 
            p = g_0
        else:
            p = g_1 - beta*p
        alpha = np.divide(np.square(norm(g_1)),np.square(norm(np.matmul(A, p))))
        x = x + alpha*p
        r = r - alpha*np.matmul(A,p)
        g = np.matmul(transpose_matrix(A), r)
    return x        
