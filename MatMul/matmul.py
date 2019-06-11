# !/usr/bin/env python
'''
Polynomial code with fast decoding
'''

from mpi4py import MPI
import numpy as np
import random
import threading
import time

# Change to True for more accurate timing, sacrificing performance
# from data_utils import load_CIFAR10

barrier = True
# Change to True to imitate straggler effects
straggling = False
timing = True


def loop():
    t = time.time()
    while time.time() < t + 60:
        a = 1 + 1


def recursive_fft(y):
    n = len(y)
    a = np.zeros(n, dtype=complex)
    if n == 1:
        return y
    w = 1
    y0 = y[::2]
    y1 = y[1::2]
    a0 = recursive_fft(y0)
    a1 = recursive_fft(y1)
    for k in range(int(n / 2)):
        a[k] = a0[k] + w * a1[k]
        a[k + int(n / 2)] = a0[k] - w * a1[k]
        w = w * np.exp(-2 * np.pi * 1j / n)
    return a


##################### Parameters ########################
# Use one master and N workers
N = 17

# Matrix division
m = 4
n = 4

# Input matrix size - A: s by r, B: s by t
s = 3000  # 3073  # 3
r = 500  # 500  # 8
t = 500  # 12  # 8

# CIFAR-10 constants

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Pick a primitive root 64
rt = 64

# Values of x_i used by 17 workers
var = [np.exp(2 * np.pi * i * 1j / 16) for i in range(16)] + [1]
#########################################################

comm = MPI.COMM_WORLD

loss = 0.0
dW = np.zeros((t, s))
X = np.zeros((r, s))
W = np.zeros((t, s))
y = np.zeros((r,))
reg = 0.01
learning_rate = 0.1
acc = 0


if comm.rank == 0:
    # Master
    START = time.time()
    A = np.random.randn(0, 255, (r, s))
    B = np.random.randn(0, 255, (t, s))
    print(A.shape)
    print(B.shape)
    if timing:
        print("Running with %d processes:" % comm.Get_size())

    # Decide and broadcast chosen straggler

    straggler = random.randint(1, N)
    for i in range(N):
        comm.send(straggler, dest=i + 1, tag=7)

    # Split the matrices
    Ap = np.split(A, m)
    Bp = np.split(B, n)

    # Encode the matrices
    Aenc = [sum([Ap[j] * (pow(var[i], j)) for j in range(m)]) for i in range(N)]
    Benc = [sum([Bp[j] * (pow(var[i], j * m)) for j in range(n)]) for i in range(N)]

    # Initialize return dictionary
    Rdict = []
    for i in range(N):
        Rdict.append(np.zeros((int(r / m), int(t / n)), dtype=np.complex))

    # Start requests to send and receive
    reqA = [None] * N
    reqB = [None] * N
    reqC = [None] * N

    bp_start = time.time()

    for i in range(N):
        reqA[i] = comm.Isend(Aenc[i], dest=i + 1, tag=15)
        reqB[i] = comm.Isend(Benc[i], dest=i + 1, tag=29)
        reqC[i] = comm.Irecv(Rdict[i], source=i + 1, tag=42)

    MPI.Request.Waitall(reqA)
    MPI.Request.Waitall(reqB)

    # Optionally wait for all workers to receive their submatrices, for more accurate timing
    if barrier:
        comm.Barrier()

    bp_sent = time.time()
    if timing:
        print("Time spent sending all messages is: %f" % (bp_sent - bp_start))

    Crtn = []
    for i in range(N):
        Crtn.append(np.zeros((int(r / 4), int(t / 4)), dtype=complex))

    lst = []
    # Wait for the mn fastest workers
    start1 = 0
    end1 = 0
    for i in range(m * n + 1):
        j = MPI.Request.Waitany(reqC)
        lst.append(j)
        Crtn[j] = Rdict[j]
        if i == m * n - 1:
            start1 = time.time()
        if i == m * n:
            end1 = time.time()

    if timing:
        print("straggler:")
        print(end1 - start1)

    bp_received = time.time()
    if timing:
        print("Time spent waiting for %d workers %s is: %f" % (
            m * n, ",".join(map(str, [x + 1 for x in lst])), (bp_received - bp_sent)))

    missing = set(range(m * n)) - set(lst)

    # Calculate missing

    for l in missing:
        Crtn[l] = np.zeros(np.shape(Crtn[0]), dtype=np.complex)
        Crtn[l] += np.dot(Ap[0], Bp[0].T) * 16
        for i in range(16):
            if not i == l:
                Crtn[l] -= Crtn[i]

    # Fast decoding hard coded for m, n = 4

    Cres = np.zeros((r, t), dtype=complex)
    jump_x = int(r / 4)
    jump_y = int(t / 4)
    for i in range(jump_x):
        for j in range(jump_y):
            value = []
            for k in range(16):
                value.append(Crtn[k][i, j])
            coeff = recursive_fft(value) / 16
            for k_y in range(4):
                for k_x in range(4):
                    Cres[i + k_x * jump_x, j + k_y * jump_y] = coeff[k_x + 4 * k_y]

    bp_done = time.time()
    if timing:
        print("Time spent decoding is: %f" % (bp_done - bp_received))

    END = time.time()
    print("TIME:")
    print(END-START)


else:
    # Worker
    # Receive straggler information from the master
    straggler = comm.recv(source=0, tag=7)

    # Receive split input matrices from the master
    Ai = np.empty_like(np.matrix([[0.0 + 0.0j] * s for i in range(int(r / m))]))
    Bi = np.empty_like(np.matrix([[0.0 + 0.0j] * s for i in range(int(t / n))]))

    rA = comm.Irecv(Ai, source=0, tag=15)
    rB = comm.Irecv(Bi, source=0, tag=29)

    rA.wait()
    rB.wait()

    Ai = np.matrix(Ai)
    Bi = np.matrix(Bi)

    if barrier:
        comm.Barrier()
    wbp_received = time.time()

    # Start a separate thread to mimic background computation tasks if this is a straggler
    if straggling:
        if straggler == comm.rank:
            t = threading.Thread(target=loop)
            t.start()

    Ci = (Ai * Bi.T)

    wbp_done = time.time()
    if timing:
        pass
        print("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()

