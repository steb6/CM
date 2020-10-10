import numpy as np
#from utils import norm
from numpy.linalg import norm
from data_manager import read_data
import time
import tqdm
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from conjugate_gradient import conjugate_gradient


A, b = read_data('data/ML-CUP19-TR.csv')
m, n = A.shape

# Our solution
start = time.monotonic_ns()
x, status, ite = conjugate_gradient(A, b)
done = time.monotonic_ns()
elapsed = done - start
print("our implementation: ns spent: ", elapsed)
print("status: ", status)
print("iterations: ", ite)
print("||Ax - b|| = ", norm(np.matmul(A, x) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, x) - b), norm(b)))
print("||A*Ax - A*b|| =", norm(np.matmul(np.transpose(A),np.matmul(A, x)) - np.matmul(np.transpose(A),b)))


Q, R = np.linalg.qr(A)
print("||Q^T (Ax - b)||", norm(np.matmul(Q.T, np.matmul(A, x) - b)))

# Library Leas Squares solution
start = time.monotonic_ns()
xnp = np.linalg.lstsq(A, b, rcond=None)
done = time.monotonic_ns()
elapsed = done - start
print("numpy.linalg.qr: ns spent: ", elapsed)
print("||Ax - b|| =", norm(np.matmul(A, xnp[0]) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, xnp[0]) - b), norm(b)))
print("||A*Ax - A*b|| =", norm(np.matmul(np.transpose(A),np.matmul(A, x)) - np.matmul(np.transpose(A),b)))


cg_tries = []
for count in tqdm.tqdm(range(1000)):
    start = time.monotonic_ns()
    x, status, ite = conjugate_gradient(A, b)
    done = time.monotonic_ns()
    elapsed = done - start
    cg_tries.append(elapsed)
cg_tries = np.array(cg_tries)
cg_tries.sort()
cg_tries = cg_tries[:-100]
cg_tries = cg_tries[100:]
print("Mean elapsed time over 1000 tries: ",  cg_tries.mean())

acc = []
for exponent in range(-11, 0):
    acc.append(10**exponent)
acc.reverse()

A_ = np.random.rand(m, n)
b_ = np.random.rand(m)
r_min = A_.min()
r_max = A_.max()
b_min = b_.min()
b_max = b_.max()
A_ = (((A_ - r_min) / (r_max - r_min)) * (A.max() - A.min())) + A.min()
b_ = (((b_ - b_min) / (b_max - b_min)) * (b.max() - b.min())) + b.min()
iterations = []
iterations_ = []
for eps in acc:
    x, status, ite = conjugate_gradient(A, b, eps = eps, maxIter=1000000)
    x_, status_, ite_ = conjugate_gradient(A_, b_, eps = eps, maxIter=1000000)
    iterations.append(ite)
    iterations_.append(ite_)
    
# Creating plot
print("Creating plot...")
plt.plot(iterations, acc)
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")
plt.yscale('log',basey=10) 
plt.savefig("results/cg_accuracy.png")
plt.show()
plt.plot(iterations_, acc)
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")
plt.yscale('log',basey=10) 
plt.savefig("results/cg_accuracy_rand.png")
plt.show()