#!/usr/bin/env python
'''
Polynomial code with fast decoding
'''

from mpi4py import MPI
import numpy as np
import random
import threading
import time
from data_utils import load_CIFAR10

# Change to True for more accurate timing, sacrificing performance
barrier = True
# Change to True to imitate straggler effects
straggling = False


def loop():
    t = time.time()
    while time.time() < t + 60:
        a = 1 + 1


##################### Parameters ########################
# Use one master and N workers
N = 17

# Matrix division
m = 4
n = 4

# Field size assumed to be prime for this implementation
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

F = 65537
s = 3073
r = num_dev
t = 10

# Pick a primitive root 64
rt = 64

# Values of x_i used by 17 workers
var = [pow(64, i, 65537) for i in range(16)] + [3]
#########################################################

comm = MPI.COMM_WORLD

if comm.rank == 0:

    # Decide and broadcast chose straggler
    straggler = random.randint(1, N)
    for i in range(N):
        comm.send(straggler, dest=i + 1, tag=7)

    cifar10_dir = 'Datasets/cifar-10-batches-py'

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    A = X_dev
    B = (np.random.randn(3073, 10) * 0.0001).T

    if not r % m == 0:
        new_r = r + m - r % m
        A = np.pad(A, ((0, 0), (0, new_r - r)), 'constant')
        r = new_r
    if not t % n == 0:
        new_t = t + n - t % n
        B = np.pad(B, ((0, 0), (0, new_t - t)), 'constant')
        t = new_t


    # Split the matrices
    Ap = np.split(A, m)
    Bp = np.split(B, n)

    # Encode the matrices
    Aenc = [sum([Ap[j] * (pow(var[i], j, F)) for j in range(m)]) % F for i in range(N)]
    Benc = [sum([Bp[j] * (pow(var[i], j * m, F)) for j in range(n)]) % F for i in range(N)]

    # Initialize return dictionary
    Rdict = []
    for i in range(N):
        Rdict.append(np.zeros((int(r / m), int(t / n)), dtype=np.int_))

    # Start requests to send and receive
    reqA = [None] * N
    reqB = [None] * N
    reqC = [None] * N

    bp_start = time.time()

    for i in range(N):
        reqA[i] = comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
        reqB[i] = comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
        reqC[i] = comm.Irecv([Rdict[i], MPI.INT], source=i + 1, tag=42)

    MPI.Request.Waitall(reqA)
    MPI.Request.Waitall(reqB)

    # Optionally wait for all workers to receive their submatrices, for more accurate timing
    if barrier:
        comm.Barrier()

    bp_sent = time.time()

    Crtn = [None] * N
    lst = []
    # Wait for the mn fastest workers
    for i in range(m * n):
        j = MPI.Request.Waitany(reqC)
        lst.append(j)
        Crtn[j] = Rdict[j]
    bp_received = time.time()

    missing = set(range(m * n)) - set(lst)

    # Fast decoding hard coded for m, n = 4
    sig = 4
    xlist = [var[i] for i in lst]

    for i in missing:
        begin = time.time()
        coeff = [1] * (m * n)
        for j in range(m * n):
            # Compute coefficient
            for k in set(lst) - set([lst[j]]):
                coeff[j] = (coeff[j] * (var[i] - var[k]) * pow(var[lst[j]] - var[k], F - 2, F)) % F
        Crtn[i] = sum([Crtn[lst[j]] * coeff[j] for j in range(16)]) % F

    for k in range(4):
        jump = 2 ** (3 - k)
        for i in range(int(jump)):
            block_num = 8 / jump
            for j in range(int(block_num)):
                base = i + j * jump * 2
                Crtn[base] = ((Crtn[base] + Crtn[base + jump]) * 32769) % F
                Crtn[int(base + jump)] = ((Crtn[int(base)] - Crtn[int(base + jump)]) * var[int((-i * block_num) % 16)]) % F
    bp_done = time.time()
    print('c:')
    print(Crtn)
    print('dot:')
    print(np.dot(A, B))
    print('equal?')
    print(Crtn == np.dot(A, B))

    # Verify correctness
    # Bit reverse the order to match the FFT
    # To obtain outputs in an ordinary order, bit reverse the order of input matrices prior to FFT
    # bit_reverse = [0, 2, 1, 3]
    # Cver = [(Ap[bit_reverse[i / 4]] * Bp[bit_reverse[i % 4]].getT()) % F for i in range(m * n)]
    # print ([np.array_equal(Crtn[i], Cver[i]) for i in range(m * n)])
else:
    # Worker
    # Receive straggler information from the master
    straggler = comm.recv(source=0, tag=7)

    # Receive split input matrices from the master
    Ai = np.empty_like(np.matrix([[0] * s for i in range(int(r / m))]))
    Bi = np.empty_like(np.matrix([[0] * s for i in range(int(t / n))]))
    rA = comm.Irecv(Ai, source=0, tag=15)
    rB = comm.Irecv(Bi, source=0, tag=29)

    rA.wait()
    rB.wait()

    if barrier:
        comm.Barrier()
    wbp_received = time.time()

    # Start a separate thread to mimic background computation tasks if this is a straggler
    if straggling:
        if straggler == comm.rank:
            t = threading.Thread(target=loop)
            t.start()

    Ci = (Ai * (Bi.getT())) % F
    wbp_done = time.time()
    # print "Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received)

    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()
