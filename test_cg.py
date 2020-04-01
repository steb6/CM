from qr_householder import qr_factorization1, qr_factorization2, qr_factorization3
import numpy as np
from utils import norm
from data_manager import read_data
import time
from conjugate_gradient import conjugate_gradient

functions = [qr_factorization1, qr_factorization2, qr_factorization3]

A, b = read_data('data/ML-CUP19-TR.csv')
m, n = A.shape
# x = conjugate_gradient(A, b)
