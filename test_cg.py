import numpy as np
#from utils import norm
from numpy.linalg import norm
from data_manager import read_data
import time
from conjugate_gradient import conjugate_gradient

A, b = read_data('data/ML-CUP19-TR.csv')
m, n = A.shape

# PARAMETERS TO EXPERIMENTAL SET UP: initial guess x0, beta, stopping condition so epsilon and maxiteration(?) 

# Our solution
start = time.monotonic_ns()
x, status = conjugate_gradient(A, b)
done = time.monotonic_ns()
elapsed = done - start
print("our implementation: ns spent: ", elapsed)
print("status: ", status)
print("||Ax - b|| = ", norm(np.matmul(A, x) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x) - b), norm(b)))
print("||A*Ax - A*b|| =", norm(np.matmul(np.matmul(np.transpose(A),A),x) - np.matmul(np.transpose(A),b)))
# Library Leas Squares solution
start = time.monotonic_ns()
xnp = np.linalg.lstsq(A, b, rcond=None)
done = time.monotonic_ns()
elapsed = done - start
print("numpy.linalg.qr: ns spent: ", elapsed)
print("||Ax - b|| =", norm(np.matmul(A, xnp[0]) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, xnp[0]) - b), norm(b)))