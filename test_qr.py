from qr_householder import qr_factorization3
import numpy as np
from numpy.linalg import norm
from data_manager import read_data
import time
import matplotlib.pyplot as plt
import tqdm
from numpy.linalg import lstsq
from utils import conditioning_angle
from utils import row_to_column, column_to_row

# Parameters:
# Number of tries to compute mean
TRIES = 20
# Number of first/last elements to remove from tries
CUT = 4

# Read data
A, b = read_data('data/ML-CUP19-TR.csv', add_augmented_columns=True)
m, n = A.shape

# Properties of the matrix A
print("********** PROPERTIES OF THE MATRIX A **********")
print('Dim: {} x {}'.format(m, n))
print("Min: {}, Max: {}".format(A.min(), A.max()))
print("Rank of A is " + str(np.linalg.matrix_rank(A)))
print("Condition number of matrix A is " + str(np.linalg.cond(A)))
check = np.random.rand(A.shape[0], A.shape[1])
c_min = check.min()
c_max = check.max()
check = (((check - c_min) / (c_max - c_min)) * (A.max() - A.min())) + A.min()
assert np.isclose(check.min(), A.min()) and np.isclose(check.max(), A.max())
print("Condition number of random matrix with same dimension and range of values of A is " + str(np.linalg.cond(check)))

# Compute solution
print("********** COMPUTE SOLUTION **********")
# Library solution
start = time.monotonic_ns()
Q, R = np.linalg.qr(A)
done = time.monotonic_ns()
elapsed = done - start
print("numpy.linalg.qr: ns spent: ", elapsed)
x_np = np.matmul(np.linalg.inv(R), np.matmul(Q.T, b))

# Our solution
start = time.monotonic_ns()
V, R = qr_factorization3(A)
done = time.monotonic_ns()
elapsed = done - start
# Ax = b <=> QRx = b <=> x = R^{-1} Q^{T} b
# Compute Q^{T} b
x = np.copy(b)
for j, vi in enumerate(V):
    aux1 = np.matmul(column_to_row(vi), x[j:]) # vi^T * b
    vi.shape = (vi.shape[0], 1)
    aux2 = 2*np.matmul(vi, aux1) # 2*vi*vi^T
    x[j:] = x[j:] - aux2 # b - 2*vi*vi^T*b

R = R[:n, :]
R_inv = np.linalg.inv(R)
x = np.matmul(R_inv, x[:R_inv.shape[1]])

assert np.isclose(norm(x-x_np), 0, atol=1.e-5)

# print("||A - QR|| =", norm(A - np.matmul(Q1, R1)))
# print("||A - QR||/||A||", np.divide(norm(A - np.matmul(Q, R)), norm(A)))
# print("||Ax - b|| =", norm(np.matmul(A, x) - b))
print("our implementation: ns spent: ", elapsed)
print("||Ax - b|| = ", norm(np.matmul(A, x) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x) - b), norm(b)))
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x) - b)))
print("Conditioning angle: ", conditioning_angle(A, b, x))

# Check if computational cost scale with m
print("********** Checking computational cost **********")
times = []
sizes = range(n, m, 100)
for k in sizes:
    A_ = A[:k, :]
    tries = []
    for i in range(TRIES):
        start = time.monotonic_ns()
        V, R = qr_factorization3(A_)
        done = time.monotonic_ns()
        elapsed = done - start
        tries.append(elapsed)
    tries = np.array(tries)
    tries.sort()
    tries = tries[:-CUT]
    tries = tries[CUT:]
    times.append(tries.mean())

# Creating plot
print("Creating plot...")
plt.plot(sizes, times)
x = np.array(sizes)
y = np.array(times)
A = np.vstack([x, np.ones(len(x))]).T
m, c = lstsq(A, y, rcond=None)[0]
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.ylabel("Time for QR factorization")
plt.xlabel("Largest dimension of A")
plt.title("ML-cup matrix")
plt.show()


# Now with m>>n
print("m>>n: Factorizing for various m...")
times = []
n = 5
m_init = 10000
m_end = 50000
m_step = 2500
sizes = range(m_init, m_end, m_step)
for k in tqdm.tqdm(sizes):
    A_ = np.random.rand(k, n)
    tries = []
    for i in range(TRIES):
        start = time.monotonic_ns()
        V, R = qr_factorization3(A_)
        done = time.monotonic_ns()
        elapsed = done - start
        tries.append(elapsed)
    tries = np.array(tries)
    tries.sort()
    tries = tries[:-CUT]
    tries = tries[CUT:]
    times.append(tries.mean())

# Creating plot
print("Creating plot...")
times = [elem/1000000 for elem in times]
x = np.array(sizes)
y = np.array(times)
A = np.vstack([x, np.ones(len(x))]).T
m, c = lstsq(A, y, rcond=None)[0]
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.plot(sizes, times, label='Times')
plt.legend()
plt.ylabel("Time for QR factorization")
plt.xlabel("Largest dimension of A")
plt.title("Random matrix")
plt.show()
