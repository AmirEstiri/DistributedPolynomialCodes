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
from data_utils import load_CIFAR10

barrier = True
# Change to True to imitate straggler effects
straggling = False


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
s = 6000 # 3073  # 3
r = 6000  # 500  # 8
t = 12  # 12  # 8

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

if comm.rank == 0:
    # Master

    cifar10_dir = 'Datasets/cifar-10-batches-py'

    # X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    #
    # mask = range(num_training, num_training + num_validation)
    # X_val = X_train[mask]
    # y_val = y_train[mask]
    #
    # mask = range(num_training)
    # X_train = X_train[mask]
    # y_train = y_train[mask]
    #
    # mask = np.random.choice(num_training, num_dev, replace=False)
    # X_dev = X_train[mask]
    # y_dev = y_train[mask]
    #
    # mask = range(num_test)
    # X_test = X_test[mask]
    # y_test = y_test[mask]
    #
    # X_train = np.reshape(X_train, (X_train.shape[0], -1))
    # X_val = np.reshape(X_val, (X_val.shape[0], -1))
    # X_test = np.reshape(X_test, (X_test.shape[0], -1))
    # X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    #
    # mean_image = np.mean(X_train, axis=0)
    # X_train -= mean_image
    # X_val -= mean_image
    # X_test -= mean_image
    # X_dev -= mean_image
    #
    # X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    # X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    # X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    print("Running with %d processes:" % comm.Get_size())

    # Decide and broadcast chose straggler

    # double = np.random.randn(2, 2)
    # double = np.array(double)
    # print(double)
    # test_req = comm.Isend(double, dest=1, tag=19)
    # test_req.wait()

    straggler = random.randint(1, N)
    for i in range(N):
        comm.send(straggler, dest=i + 1, tag=7)


    # A = np.array(
    #     [[-28.2, -87, -99], [-10, 22, -16], [87, 11, 17], [10, 90, 55], [54, 57, 91],
    #      [44, 74, 96],
    #      [52, 28, 11],
    #      [12, 99, 0]], dtype=complex)
    # B = np.array(
    #     [[246.00, 194, 225], [255, 196, 194], [155, 116, 112], [87, 13, 24], [4, 78, 134], [29, 215, 36],
    #      [148, 88, 232],
    #      [195, 160, 202]], dtype=complex)

    # A = X_dev
    # B = np.random.randn(12, 3073) * 0.0001

    # A = np.random.randn(r, s) * 0.0001
    # B = np.random.randn(t, s) * 0.0001

    A = np.matrix(np.random.random_integers(0, 255, (r, s)))
    B = np.matrix(np.random.random_integers(0, 255, (t, s)))

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

    # comm.Isend(Aenc[0], dest=1, tag=23)

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
    print("Time spent sending all messages is: %f" % (bp_sent - bp_start))

    # Crtn = [None] * N
    Crtn = []
    for i in range(N):
        Crtn.append(np.zeros((int(r/4), int(t/4)), dtype=complex))

    lst = []
    # Wait for the mn fastest workers
    start1 = 0
    end1 = 0
    for i in range(m * n + 1):
        j = MPI.Request.Waitany(reqC)
        lst.append(j)
        Crtn[j] = Rdict[j]
        if i == m*n - 1:
            start1 = time.time()
        if i == m*n:
            end1 = time.time()

    print("straggler:")
    print(end1 - start1)

    bp_received = time.time()
    print("Time spent waiting for %d workers %s is: %f" % (
        m * n, ",".join(map(str, [x + 1 for x in lst])), (bp_received - bp_sent)))

    missing = set(range(m * n)) - set(lst)
    # print(lst)
    # print(missing)
    # print(Crtn)

    # Fast decoding hard coded for m, n = 4

    # for l in missing:
    #     pstar = l % 4
    #     kstar = int((l - pstar) / 4)
    #     Crtn[l] = np.dot(Ap[pstar], Bp[kstar].T)
    #     Crtn[l] = np.array(Crtn[l])
    #     for i in range(16):
    #         if not i == l:
    #             Crtn[l] -= Crtn[i] * np.exp(-2 * np.pi * i * 1j * l / n)
    #     Crtn[l] *= np.exp(2 * np.pi * 1j * l * l / n)
    #
    #
    # ###########
    #
    # l = 3
    # pstar = l % 4
    # kstar = int((l - pstar) / 4)
    # Crtn_test = np.dot(Ap[pstar], Bp[kstar].T)
    # Crtn_test = np.array(Crtn[l])
    # for i in range(16):
    #     if not i == l:
    #         Crtn_test -= Crtn[i] * np.exp(-2 * np.pi * i * 1j * l / n)
    # Crtn_test *= np.exp(2 * np.pi * 1j * l * l / n)
    # print(Crtn_test)
    # print(Crtn[3])
    #
    # ##########
    # print(Crtn)

    Cres = np.zeros((r, t), dtype=complex)
    jump_x = int(r/4)
    jump_y = int(t/4)
    for i in range(jump_x):
        for j in range(jump_y):
            value = []
            for k in range(16):
                value.append(Crtn[k][i, j])
            coeff = recursive_fft(value) / 16
            for k_y in range(4):
                for k_x in range(4):
                    Cres[i + k_x*jump_x, j + k_y*jump_y] = coeff[k_x + 4 * k_y]

    bp_done = time.time()
    print("Time spent decoding is: %f" % (bp_done - bp_received))

    print('C:')
    print(Cres)
    print('dot:')
    dot = np.dot(A, B.T)
    print(dot)
    # print('divided')
    # print(Cres / dot)
    # print('equal?')
    # print(Crtn == np.dot(A, B.T))


else:
    # Worker
    # Receive straggler information from the master
    straggler = comm.recv(source=0, tag=7)

    # if comm.rank == 1:
    #     # double = np.matrix(np.zeros((2, 2)))
    #     double = np.zeros((2, 2))
    #     test = comm.Irecv(double, source=0, tag=19)
    #     test.wait()
    #     print(double)

    # Receive split input matrices from the master
    Ai = np.empty_like(np.matrix([[0.0 + 0.0j] * s for i in range(int(r / m))]))
    Bi = np.empty_like(np.matrix([[0.0 + 0.0j] * s for i in range(int(t / n))]))

    # Ai = np.zeros((int(r / m), s))
    # Bi = np.zeros((int(t / n), s))
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
    # print "Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received)

    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()
