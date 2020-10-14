import tqdm
import numpy as np
import time
from qr_method import qr_method
from conjugate_gradient import conjugate_gradient
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# Parameters
TRIES = 10
CUT = 3

# Type of matrices, remove from here to jump some tests
MATRICES = ["square", "little_m", "big_m"]

# Iterate over type of matrices and get right dimensions
for matrix in MATRICES:
    if matrix is "square":
        n = 0
        m_init = 10
        m_end = 1000
        m_step = 50
    elif matrix is "little_m":
        n = 50
        m_init = 50
        m_end = 500
        m_step = 5
    elif matrix is "big_m":
        n = 5
        m_init = 500
        m_end = 50000
        m_step = 500
    else:
        n = 0
        m_init = 0
        m_end = 0
        m_step = 0
        exit()
    # Set up range of dimensions
    print("Testing " + matrix + " matrix")
    times_qr = []
    times_cg = []
    sizes = range(m_init, m_end, m_step)
    # For every dimensions to test
    for k in tqdm.tqdm(sizes):
        # Select square or tall thin matrix and b
        if matrix is "square":
            A_ = np.random.rand(k, k)
        else:
            A_ = np.random.rand(k, n)
        b_ = np.random.rand(k)
        # In order to have smoother lines, compute TRIES times and cut first and last CUT ones for QR
        tries = []
        for _ in range(TRIES):
            start = time.perf_counter_ns()
            _ = qr_method(A_, b_)
            done = time.perf_counter_ns()
            elapsed = done - start
            tries.append(elapsed)
        tries = np.array(tries)
        tries.sort()
        tries = tries[:-CUT]
        tries = tries[CUT:]
        times_qr.append(tries.mean())
        # Do the same with CG
        tries_cg = []
        for _ in range(TRIES):
            start = time.perf_counter_ns()
            _, _, _ = conjugate_gradient(A_, b_)
            done = time.perf_counter_ns()
            elapsed = done - start
            tries_cg.append(elapsed)
        tries_cg = np.array(tries_cg)
        tries_cg.sort()
        tries_cg = tries_cg[:-CUT]
        tries_cg = tries_cg[CUT:]
        times_cg.append(tries_cg.mean())

    # Save sizes and times for both methods
    np.savetxt("results/qr_" + matrix + ".txt", (sizes, times_qr))
    np.savetxt("results/cg_" + matrix + ".txt", (sizes, times_cg))

    # Creating plot
    print("Creating plot...")
    times_qr = [elem / 1000000 for elem in times_qr]
    times_cg = [elem / 1000000 for elem in times_cg]
    # Fitted line only for the square one
    if matrix is not "square":
        x = np.array(sizes)
        y = np.array(times_qr)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = lstsq(A, y, rcond=None)[0]
        plt.plot(x, m * x + c, 'r', label='Fitted line for QR times')
    plt.plot(sizes, times_qr, label='QR Times')
    plt.plot(sizes, times_cg, 'g', label='CG Times')
    plt.legend()
    plt.ylabel("Times to compute x")
    # Get right x label
    if matrix is "square":
        plt.xlabel("Dimensions of A")
    else:
        plt.xlabel("Largest dimension of A")
    plt.title(matrix)
    plt.savefig("results/" + matrix + ".png")
    plt.clf()
