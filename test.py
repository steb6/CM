from qr_householder import qr_factorization1, qr_factorization2, qr_factorization3
import numpy as np
from utils import norm
from data_manager import read_data
import time


A = read_data('data/ML-CUP19-TR.csv')
start = time.monotonic_ns()
Q, R = qr_factorization1(A)
done = time.monotonic_ns()
elapsed = done - start
print("ns spent: ", elapsed)
# R = np.around(R, decimals=6)
print("QR Factorization ended")
print("||A - QR|| =", norm(A - np.matmul(Q, R)))

start = time.monotonic_ns()
Q, R = qr_factorization2(A)
done = time.monotonic_ns()
elapsed = done - start
print("ns spent: ", elapsed)
print("QR Factorization ended")
print("||A - QR|| =", norm(A - np.matmul(Q, R)))

start = time.monotonic_ns()
Q, R = qr_factorization3(A)
done = time.monotonic_ns()
elapsed = done - start
print("ns spent: ", elapsed)
print("QR Factorization ended")
print("||A - QR|| =", norm(A - np.matmul(Q, R)))
