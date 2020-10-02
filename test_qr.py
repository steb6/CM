from qr_householder import qr_factorization1, qr_factorization2, qr_factorization3
import numpy as np
# from utils import norm
from numpy.linalg import norm
from data_manager import read_data
import time
import matplotlib.pyplot as plt
import tqdm
from numpy.linalg import lstsq
from utils import conditioning_angle
from utils import row_to_column, column_to_row
from numpy.linalg import qr
from scipy.linalg import qr
functions = [qr_factorization3]
# functions = [qr_factorization3]


A, b = read_data('data/ML-CUP19-TR.csv', add_augmented_columns=False)
m, n = A.shape

for i, qr_factorization in enumerate(functions):
    print("**********************************************")
    print(qr_factorization.__name__)
    print('Readed {} rows and {} column'.format(A.shape[0], A.shape[1]))
    amin = A.min()
    amax = A.max()
    print("min of A: {}, max of A: {}".format(A.min(), A.max()))
    print("Rank of A is " + str(np.linalg.matrix_rank(A)))
    print("Condition number of matrix A is "+str(np.linalg.cond(A)))
    check = np.random.rand(A.shape[0], A.shape[1])
    cmin = check.min()
    cmax = check.max()
    check = (((check - cmin) / (cmax - cmin))*(A.max()-A.min()))+A.min()
    print("min of check: {}, max of check:{}".format(check.min(), check.max()))
    print("Condition number of random matrix with same dimension of A is "+str(np.linalg.cond(check)))
    start = time.monotonic_ns()
    V, R = qr_factorization(A)
    done = time.monotonic_ns()
    elapsed = done - start

    # Ax = b <=> QRx = b <=> x = R^{-1} Q^{T} b
    # Compute Q^{T} b
    b_copy = np.copy(b)
    for j, vi in enumerate(V):
        aux1 = np.matmul(column_to_row(vi), b_copy[j:]) # vi^T * b
        vi.shape = (vi.shape[0], 1)
        aux2 = 2*np.matmul(vi, aux1) # 2*vi*vi^T
        # igrande = np.matmul(np.eye(m), b_copy) # I * b
        b_copy[j:] = b_copy[j:] - aux2 # b - 2*vi*vi^T*b

    R = R[:n, :]
    R_inv = np.linalg.inv(R)
    x = np.matmul(R_inv, b_copy[:R_inv.shape[1]])

    print("Solution with implicit product: ||Ax - b|| = ", norm(np.matmul(A, x) - b))

    Q, R = np.linalg.qr(A)
    print("||A - QR|| = ", norm(A - np.matmul(Q, R)))
    # x = np.matmul(np.linalg.inv(R), np.matmul(Q.T, b))
    # print("||Ax - b|| = ", norm(np.matmul(A, x) - b))
    # exit()

    print("QR Factorization n° {} ended".format(i+1))
    print("ns spent: ", elapsed)
    # print("||A - QR|| =", norm(A - np.matmul(Q1, R1)))
    # print("||A - QR||/||A||", np.divide(norm(A - np.matmul(Q1, R1)), norm(A)))
    print("||Ax - b|| =", norm(np.matmul(A, x) - b))
    print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x) - b), norm(b)))
    print("Conditioning angle: ", conditioning_angle(A, b, x))
    # Provo con il problema perturbato, per vedere se è un algoritmo stabile (lo è)
    # delta = np.random.rand(b.shape[0], 1)
    # delta = np.array(map(lambda x_: 1 if x_ < 0.5 else -1, delta))
    # btilde = b + np.random.randint(-np.finfo(float).eps, np.finfo(float), b.shape, float)

# Check if computational cost scale with m
#print("**********************************************************")
#print("Factorizing for various m...")
#times = []
#sizes = range(n, m, 100)
#for k in sizes:
#    print(k/m)
#    A_ = A[:k, :]
#    start = time.monotonic_ns()
#    Q, R = qr_factorization(A_)
#    done = time.monotonic_ns()
#    elapsed = done - start
#    times.append(elapsed)

# Creating plot
#print("Creating plot...")
#plt.plot(sizes, times)
#plt.ylabel("Time for QR factorization")
#plt.xlabel("Largest dimension of A")
#plt.show()

#exit()

# Now with m>>n
print("**********************************************************")
print("m>>n: Factorizing for various m...")
times = []
n = 5
m_init = 10000
m_end = 50000
m_step = 5000
sizes = range(m_init, m_end, m_step)
for k in tqdm.tqdm(sizes):
    # print("Biggest dimension is "+str(k))
    A_ = np.random.rand(k, n)
    start = time.monotonic_ns()
    V, R = qr_factorization3(A_)
    # Q, R = np.linalg.qr(A_)
    done = time.monotonic_ns()
    elapsed = done - start
    times.append(elapsed)

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
plt.show()
