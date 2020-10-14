import numpy as np
from numpy.linalg import norm
from data_manager import read_data
import time
import tqdm
import matplotlib.pyplot as plt
from conjugate_gradient import conjugate_gradient
import random

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

# Library Least Squares solution
start = time.monotonic_ns()
xnp = np.linalg.lstsq(A, b, rcond=None)
done = time.monotonic_ns()
elapsed = done - start
print("numpy.linalg.qr: ns spent: ", elapsed)
print("||Ax - b|| =", norm(np.matmul(A, xnp[0]) - b))
print("||Ax - b||/||b|| =", np.divide(norm(np.matmul(A, xnp[0]) - b), norm(b)))
print("||A*Ax - A*b|| =", norm(np.matmul(np.transpose(A),np.matmul(A, x)) - np.matmul(np.transpose(A),b)))

# Execution time over 1000 tries
cg_tries = []
for count in tqdm.tqdm(range(1000)):
    start = time.monotonic_ns()
    x, _, _ = conjugate_gradient(A, b)
    done = time.monotonic_ns()
    elapsed = done - start
    cg_tries.append(elapsed)
cg_tries = np.array(cg_tries)
cg_tries.sort()
cg_tries = cg_tries[:-100]
cg_tries = cg_tries[100:]
print("Mean elapsed time over 1000 tries: ",  cg_tries.mean())

# How CG converges with different eps
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
    x, _, ite = conjugate_gradient(A, b, eps = eps, maxIter=1000000)
    x_, _, ite_ = conjugate_gradient(A_, b_, eps = eps, maxIter=1000000)
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

# Different initial starting points
xcg, _, _ = conjugate_gradient(A, b, eps = 1.e-11)
xzeros = np.zeros(A.shape[1])                      
xrand = [x+random.randint(-5,5) for x in xnp[0]]   
xrand1 = [x*random.randint(-10,10) for x in xnp[0]]
xrand2 = [random.uniform(xcg.min(), xcg.max()) for _ in range(A.shape[1])]
xrand3 = [random.uniform(xcg.min()*-10, xcg.max()*10) for _ in range(A.shape[1])]
xrand4 = [random.uniform(xcg.min()*-20, xcg.max()*20) for _ in range(A.shape[1])]
xrand5 = [random.uniform(xcg.min()*-30, xcg.max()*30) for _ in range(A.shape[1])]
x0s = [xnp[0], xcg, xrand, xzeros, xrand1, xrand2]

norms = []
iterations = []
for x0 in x0s:
    #print("Starting point: ", x0)
    diff = norm(xnp[0]-x0)
    norms.append(diff)
    print("||x - x0||", diff)
    x, _, ite = conjugate_gradient(A, b, x0, eps = 1.e-11, maxIter=1000000)
    iterations.append(ite)
    print("iterations: ", ite)
    
norms, iterations = zip(*sorted(zip(norms, iterations)))
norms, iterations = (list(t) for t in zip(*sorted(zip(norms, iterations))))
# Creating plot
print("Creating plot...")
plt.plot(norms, iterations)
i = norms.index(norm(xnp[0]-np.zeros(A.shape[1])))
plt.plot(norms[i], iterations[i], 'r*')
plt.xlabel("||x-x0||")
plt.ylabel("Iterations")
plt.savefig("results/cg_x0_ite.png")
plt.show()

