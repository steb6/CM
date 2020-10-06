import tqdm
import numpy as np
import time
from qr_householder import qr_method
from conjugate_gradient import conjugate_gradient
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import os

# Parameters
TRIES = 20
CUT = 4

# Now with m=n
if not os.path.isfile("results/square.npy"):
    print("SQUARE: Solving for various n...")
    times = []
    times_cg = []
    m_init = 10
    m_end = 2000
    m_step = 50
    sizes = range(m_init, m_end, m_step)
    for k in tqdm.tqdm(sizes):
        A_ = np.random.rand(k, k)
        b_ = np.random.rand(k)
        start = time.monotonic_ns()
        x = qr_method(A_, b_)
        done = time.monotonic_ns()
        elapsed = done - start
        times.append(elapsed)

        start = time.monotonic_ns()
        x = conjugate_gradient(A_, b_)
        done = time.monotonic_ns()
        elapsed = done - start
        times_cg.append(elapsed)        

    with open("results/square.npy", "wb") as f:
        np.save(f, sizes)
        np.save(f, times)

    # Creating plot
    print("Creating plot...")
    times = [elem/1000000 for elem in times]
    times_cg = [elem/1000000 for elem in times_cg]
    x = np.array(sizes)
    y = np.array(times)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = lstsq(A, y, rcond=None)[0]
    plt.plot(x, m*x + c, 'r', label='Fitted line')
    plt.plot(sizes, times, label='QR Times')
    plt.plot(sizes, times_cg, 'g', label='CG Times')
    plt.legend()
    plt.ylabel("Time for QR and CG methods")
    plt.xlabel("Dimension n of square matrix A nxn")
    plt.title("Random matrix")
    plt.savefig("results/square.png")
    plt.clf()

# Now with m>n ********************************************************************************************************
if not os.path.isfile("results/little_m.npy"):
    print("m>n: Solving for various m...")
    times = []
    times_cg = []
    n = 50
    m_init = 50
    m_end = 500
    m_step = 5
    sizes = range(m_init, m_end, m_step)
    for k in tqdm.tqdm(sizes):
        tries = []
        A_ = np.random.rand(k, n)
        b_ = np.random.rand(k)
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
        
        tries_cg = []
        for i in range(TRIES):
            start = time.monotonic_ns()
            x, status = conjugate_gradient(A_, b_)
            done = time.monotonic_ns()
            elapsed = done - start
            tries_cg.append(elapsed)
        tries_cg = np.array(tries_cg)
        tries_cg.sort()
        tries_cg = tries_cg[:-CUT]
        tries_cg = tries_cg[CUT:]
        times_cg.append(tries_cg.mean())

    with open("results/little_m.npy", "wb") as f:
        np.save(f, sizes)
        np.save(f, times)
        np.save(f, times_cg)

    # Creating plot
    print("Creating plot...")
    times = [elem/1000000 for elem in times]
    times_cg = [elem/1000000 for elem in times_cg]
    x = np.array(sizes)
    y = np.array(times)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = lstsq(A, y, rcond=None)[0]
    plt.plot(x, m*x + c, 'r', label='Fitted line')
    plt.plot(sizes, times, label='QR Times')
    plt.plot(sizes, times_cg, 'g', label='CG Times')
    plt.legend()
    plt.ylabel("Time for QR and CG methods")
    plt.xlabel("Largest dimension of A with m>n")
    plt.title("Random matrix")
    plt.savefig("results/little_m.png")
    plt.clf()

# Now with m>>n *******************************************************************************************************
if not os.path.isfile("results/big_m.npy"):
    print("m>>n: Solving for various m...")
    times = []
    times_cg = []
    n = 5
    m_init = 500
    m_end = 50000
    m_step = 500
    sizes = range(m_init, m_end, m_step)
    for k in tqdm.tqdm(sizes):
        A_ = np.random.rand(k, n)
        b_ = np.random.rand(k)
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
        
        tries_cg = []
        for i in range(TRIES):
            start = time.monotonic_ns()
            x, status = conjugate_gradient(A_, b_)
            done = time.monotonic_ns()
            elapsed = done - start
            tries_cg.append(elapsed)
        tries_cg = np.array(tries_cg)
        tries_cg.sort()
        tries_cg = tries_cg[:-CUT]
        tries_cg = tries_cg[CUT:]
        times_cg.append(tries_cg.mean())

    with open("results/big_m.npy", "wb") as f:
        np.save(f, sizes)
        np.save(f, times)

    # Creating plot
    print("Creating plot...")
    times = [elem/1000000 for elem in times]
    times_cg = [elem/1000000 for elem in times_cg]
    x = np.array(sizes)
    y = np.array(times)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = lstsq(A, y, rcond=None)[0]
    plt.plot(x, m*x + c, 'r', label='Fitted line')
    plt.plot(sizes, times, label='QR Times')
    plt.plot(sizes, times_cg, 'g', label='CG Times')
    plt.legend()
    plt.ylabel("Time for QR and CG methods")
    plt.xlabel("Largest dimension of A with m>>n")
    plt.title("Random matrix")
    plt.savefig("results/big_m.png")
    plt.clf()
