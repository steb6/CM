from qr_householder import qr_factorization1, qr_factorization2, qr_factorization3
import numpy as np
#from utils import norm
from numpy.linalg import norm
from data_manager import read_data
import time
from conjugate_gradient import conjugate_gradient1, conjugate_gradient2
from scipy.sparse.linalg import cg

functions = [qr_factorization1, qr_factorization2, qr_factorization3]

A, b = read_data('data/ML-CUP19-TR.csv')
m, n = A.shape
x1 = conjugate_gradient1(A, b)
x2 = conjugate_gradient2(A, b)
xnp = np.linalg.lstsq(A, b)
print("Numpy least squares: ||Ax - b|| =", norm(np.matmul(A, xnp[0]) - b))
print("CG1: ||Ax - b|| =", norm(np.matmul(A, x1) - b))
print("CG2: ||Ax - b|| =", norm(np.matmul(A, x2) - b))
