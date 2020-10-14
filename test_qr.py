from qr_method import qr_method
import numpy as np
from numpy.linalg import norm
from data_manager import read_data
import time
import matplotlib.pyplot as plt
import tqdm
from utils import conditioning_angle
from numpy.linalg import lstsq
from scipy.linalg import solve_triangular
import scipy

# Parameters:
# Number of tries to compute mean
TRIES = 20
# Number of first/last elements to remove from tries
CUT = 8

# Read data
A, b = read_data('data/ML-CUP19-TR.csv', add_augmented_columns=True)
m, n = A.shape

########################################################################################################################
# Properties of matrix A #
########################################################################################################################
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

########################################################################################################################
# NUMPY solution #
########################################################################################################################
print("********** COMPUTE SOLUTION WITH LIBRARY NUMPY**********")

sp_tries = []
x_np = None
for count in tqdm.tqdm(range(1000)):
    start = time.monotonic_ns()
    Q, R = np.linalg.qr(A)
    x_np = solve_triangular(R, np.matmul(Q.T, b))
    done = time.monotonic_ns()
    elapsed = done - start
    sp_tries.append(elapsed)
sp_tries = np.array(sp_tries)
sp_tries.sort()
sp_tries = sp_tries[:-100]
sp_tries = sp_tries[100:]

print("scipy.linalg.qr: ns spent: ", sp_tries.mean())
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x_np) - b), norm(b)))
print("||A^T(Ax - b)||", norm(np.matmul(A.T, np.matmul(A, x_np) - b)))
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x_np) - b)))
print("Space used by Q and R: ", Q.nbytes+R.nbytes)

########################################################################################################################
# SCIPY solution #
########################################################################################################################
print("********** COMPUTE SOLUTION WITH LIBRARY SCIPY**********")
sp_tries = []
x_sp = None
for count in tqdm.tqdm(range(1000)):
    start = time.monotonic_ns()
    Q, R = scipy.linalg.qr(A, mode="economic")
    x_sp = solve_triangular(R, np.matmul(Q.T, b))
    done = time.monotonic_ns()
    elapsed = done - start
    sp_tries.append(elapsed)
sp_tries = np.array(sp_tries)
sp_tries.sort()
sp_tries = sp_tries[:-100]
sp_tries = sp_tries[100:]

print("scipy.linalg.qr: ns spent: ", sp_tries.mean())
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x_sp) - b), norm(b)))
print("||A^T(Ax - b)||", norm(np.matmul(A.T, np.matmul(A, x_sp) - b)))
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x_sp) - b)))
print("Space used by Q and R: ", Q.nbytes+R.nbytes)

########################################################################################################################
# OUR solution #
########################################################################################################################
print("********** COMPUTE SOLUTION WITH OUR IMPLEMENTATION **********")
qr_tries = []
x = None
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
print("our implementation: ns spent: ", qr_tries.mean())
print("||Ax - b|| = ", norm(np.matmul(A, x) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x) - b), norm(b)))
print("||Ax - b||/(||A||*||x||)", norm(A)*norm(x))
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x) - b)))
print("Conditioning angle: ", conditioning_angle(A, b, x))
print("||A^TAx - A^TB||", norm(np.matmul(A.T, np.matmul(A, x)) - np.matmul(A.T, b)))
print("||A^T(Ax - b)||", norm(np.matmul(A.T, np.matmul(A, x) - b)))
print("||Ax - b||/||b|| * k(A)", np.divide(norm(np.matmul(A, x) - b), norm(b)) * np.linalg.cond(A))
print("||k(A)/cos(theta)", np.linalg.cond(A)/np.cos(conditioning_angle(A, b, x)))
print("||k(A) + k(A)^2 tan(theta)||",
      np.linalg.cond(A)+np.square(np.linalg.cond(A))*np.tan(conditioning_angle(A, b, x)))

########################################################################################################################
# LSTSQ solution #
########################################################################################################################
# print("********** numpy.linalg.lstsq solution **********")
# start = time.monotonic_ns()
# x_lstsq = lstsq(A, b, rcond=None)[0]
# done = time.monotonic_ns()
# elapsed = done - start
# print("numpy lstsq resolution: ns spent: ", elapsed)
# print("||Ax - b|| = ", norm(np.matmul(A, x_lstsq) - b))
# print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x_lstsq) - b), norm(b)))
# print("Conditioning angle: ", conditioning_angle(A, b, x_lstsq))

########################################################################################################################
# PSEUDOINVERSE solution #
########################################################################################################################
# print("********** Pseudoinverse solution **********")
# start = time.monotonic_ns()
# x_pi = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
# done = time.monotonic_ns()
# elapsed = done - start
# print("pseudoinverse resolution: ns spent: ", elapsed)
# print("||Ax - b|| = ", norm(np.matmul(A, x_pi) - b))
# print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x_pi) - b), norm(b)))
# print("Conditioning angle: ", conditioning_angle(A, b, x_pi))

# What if problem was well conditioned?
print("********** What if problem were well conditioned? **********")
A_ = np.random.rand(m, n)
b_ = np.random.rand(m)
# Rescale random matrix to have same range of values as ML-cup
r_min = A_.min()
r_max = A_.max()
b_min = b_.min()
b_max = b_.max()
A_ = (((A_ - r_min) / (r_max - r_min)) * (A.max() - A.min())) + A.min()
b_ = (((b_ - b_min) / (b_max - b_min)) * (b.max() - b.min())) + b.min()
x_r = qr_method(A_, b_)
print("Condition number of matrix A is " + str(np.linalg.cond(A_)))
print("||Ax - b|| = ", norm(np.matmul(A_, x_r) - b_))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A_, x_r) - b_), norm(b_)))
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
