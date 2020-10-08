from qr_householder import qr_method
import numpy as np
from numpy.linalg import norm
from data_manager import read_data
import time
import matplotlib.pyplot as plt
import tqdm
from numpy.linalg import lstsq
from utils import conditioning_angle
from psutil import virtual_memory
from numpy.linalg import lstsq

# Parameters:
# Number of tries to compute mean
TRIES = 20
# Number of first/last elements to remove from tries
CUT = 5

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
x_np = np.matmul(np.linalg.inv(R), np.matmul(Q.T, b))
done = time.monotonic_ns()
elapsed = done - start
print("numpy.linalg.qr: ns spent: ", elapsed)
# x_np = np.matmul(np.linalg.inv(R), np.matmul(Q.T, b))

# Our solution 6090093 6046052 6604075 5895389 7058815 6167695
qr_tries = []
for count in tqdm.tqdm(range(1000)):
    start = time.monotonic_ns()
    x = qr_method(A, b)
    done = time.monotonic_ns()
    elapsed = done - start
    qr_tries.append(elapsed)
qr_tries = np.array(qr_tries)
qr_tries.sort()
qr_tries = qr_tries[:-100]
qr_tries = qr_tries[100:]
print(qr_tries.mean())

assert np.isclose(norm(x-x_np), 0, atol=1.e-5)

print("our implementation: ns spent: ", elapsed)
print("||Ax - b|| = ", norm(np.matmul(A, x) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x) - b), norm(b)))
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x) - b)))
print("Conditioning angle: ", conditioning_angle(A, b, x))
print("||A^TAx - A^TB||", norm(np.matmul(A.T, np.matmul(A, x)) - np.matmul(A.T, b)))
print("||A^T(Ax - b)||", norm(np.matmul(A.T, np.matmul(A, x) - b)))
# Solution found with numpy.linalg.lstsq
print("********** numpy.linalg.lstsq solution **********")
start = time.monotonic_ns()
x_lstsq = lstsq(A, b, rcond=None)[0]
done = time.monotonic_ns()
elapsed = done - start
print("numpy lstsq resolution: ns spent: ", elapsed)
print("||Ax - b|| = ", norm(np.matmul(A, x_lstsq) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x_lstsq) - b), norm(b)))
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x_lstsq) - b)))
print("Conditioning angle: ", conditioning_angle(A, b, x_lstsq))

# Solution with pseudoinverse
print("********** Pseudoinverse solution **********")
start = time.monotonic_ns()
x_pi = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
done = time.monotonic_ns()
elapsed = done - start
print("pseudoinverse resolution: ns spent: ", elapsed)
print("||Ax - b|| = ", norm(np.matmul(A, x_lstsq) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x_lstsq) - b), norm(b)))
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x_lstsq) - b)))
print("Conditioning angle: ", conditioning_angle(A, b, x_lstsq))

# What if problem was well conditioned?
print("********** What if problem were well conditioned? **********")
A_ = np.random.rand(m, n)
b_ = np.random.rand(m)
r_min = A_.min()
r_max = A_.max()
b_min = b_.min()
b_max = b_.max()
A_ = (((A_ - r_min) / (r_max - r_min)) * (A.max() - A.min())) + A.min()
b_ = (((b_ - b_min) / (b_max - b_min)) * (b.max() - b.min())) + b.min()
x_r = qr_method(A_, b_)
# print("our implementation: ns spent: ", elapsed)
print("Condition number of matrix A is " + str(np.linalg.cond(A_)))
print("||Ax - b|| = ", norm(np.matmul(A_, x_r) - b_))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A_, x_r) - b_), norm(b_)))
# print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A_, x_r) - b_)))
print("Conditioning angle: ", conditioning_angle(A_, b_, x_r))

# Check if computational cost scale with m
print("********** Checking computational cost **********")
times = []
sizes = range(n, m, 10)
for k in tqdm.tqdm(sizes):
    A_ = A[:k, :]
    b_ = b[:k]
    tries = []
    for i in range(TRIES):
        start = time.monotonic_ns()
        x = qr_method(A_, b_)
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

exit()
# Maximum problem size, about 83'000'000
k = 10
print(virtual_memory())
for count in range(10000):
    print("Trying with k=", k)
    print(virtual_memory())
    A_ = np.random.rand(k, n)
    b_ = np.random.rand(k)
    x = qr_method(A_, b_)
    k = k*2
