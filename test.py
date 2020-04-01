from qr_householder import qr_factorization1, qr_factorization2, qr_factorization3
import numpy as np
from utils import norm
from data_manager import read_data
import time
from conjugate_gradient import conjugate_gradient

functions = [qr_factorization1, qr_factorization2, qr_factorization3]

A, b = read_data('data/ML-CUP19-TR.csv')

x = conjugate_gradient(A, b)

for i, qr_factorization in enumerate(functions):
    start = time.monotonic_ns()
    Q, R = qr_factorization(A)
    done = time.monotonic_ns()
    elapsed = done - start
    # R = np.around(R, decimals=6)
    print("QR Factorization n° {} ended".format(i+1))
    print("ns spent: ", elapsed)
    print("||A - QR|| =", norm(A - np.matmul(Q, R)))


