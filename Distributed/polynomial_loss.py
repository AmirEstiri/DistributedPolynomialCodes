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


##################### Parameters ########################
# Use one master and N workers
N = 17

# Matrix division
m = 4
n = 4

# Field size assumed to be prime for this implementation
F = 65537

# Input matrix size - A: s by r, B: s by t
s = 3  # 3073  # 3
r = 8  # 500  # 8
t = 8  # 12  # 8

# CIFAR-10 constants

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Pick a primitive root 64
rt = 64

# Values of x_i used by 17 workers
var = [pow(64, i, 65537) for i in range(16)] + [3]
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

    # A = np.matrix(np.random.random_integers(-128, 127, (r, s)))
    # A = np.random.randint(0, 255, (r, s))
    # B = np.matrix(np.random.random_integers(0, 255, (t, s)))
    # B = np.random.randint(0, 255, (t, s))
    # A = np.array([[90.28405, 32.87599, 68.99852], [69.10910, 79.22109, 23.00167], [52.87274, 117.11767, 71.17177],
    #               [101.01019, 75.90908, 50.05505],
    #               [54.54578, 57.57549, 74.09134], [22.44229, 42.74126, 74.96198], [126.52394, 96.28406, 19.11752],
    #               [5.00012, 23.99999, 94.00000]])
    # A = np.array(
    #     [[90, 32, 68], [69, 79, 23], [52, 117, 71], [101, 75, 50], [54, 57, 74], [22, 42, 74], [126, 96, 19],
    #      [5, 23, 94]])
    A = np.array(
        [[28, 87, 99], [10.1, 22, 16], [87, 11, 17], [10, 90, 55], [54, 57, 91],
         [44, 74, 96],
         [52, 28, 11],
         [12, 99, 0]])
    B = np.array(
        [[246.00, 194, 225], [255, 196, 194], [155, 116, 112], [87, 13, 24], [4, 78, 134], [29, 215, 36], [148, 88, 232],
         [195, 160, 202]])

    # A = X_dev
    # B = np.random.randn(12, 3073) * 0.0001

    # A = np.random.randn(r, s) #* 0.0001
    # B = np.random.randn(t, s) #* 0.0001

    # Split the matrices
    Ap = np.split(A, m)
    Bp = np.split(B, n)

    # Encode the matrices
    Aenc = [sum([Ap[j] * (pow(var[i], j, F)) for j in range(m)]) % F for i in range(N)]
    Benc = [sum([Bp[j] * (pow(var[i], j * m, F)) for j in range(n)]) % F for i in range(N)]

    # Initialize return dictionary
    Rdict = []
    for i in range(N):
        Rdict.append(np.zeros((int(r / m), int(t / n)), dtype=np.double))

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

    Crtn = [None] * N
    lst = []
    # Wait for the mn fastest workers
    for i in range(m * n):
        j = MPI.Request.Waitany(reqC)
        lst.append(j)
        Crtn[j] = Rdict[j]
    bp_received = time.time()
    print("Time spent waiting for %d workers %s is: %f" % (
        m * n, ",".join(map(str, [x + 1 for x in lst])), (bp_received - bp_sent)))

    missing = set(range(m * n)) - set(lst)
    # print(lst)
    # print(missing)
    # print(Crtn)

    # Fast decoding hard coded for m, n = 4
    sig = 4
    xlist = [var[i] for i in lst]


    for i in missing:
        begin = time.time()
        coeff = [1.0] * (m * n)
        for j in range(m * n):
            # Compute coefficient
            for k in set(lst) - set([lst[j]]):
                coeff[j] = (coeff[j] * (var[i] - var[k]) * pow(var[lst[j]] - var[k], F - 2, F)) % F
        Crtn[i] = sum([Crtn[lst[j]] * coeff[j] for j in range(16)]) % F

    for k in range(4):
        jump = int(2 ** (3 - k))
        for i in range(jump):
            block_num = int(8 / jump)
            for j in range(block_num):
                base = i + j * jump * 2
                Crtn[base] = ((Crtn[base] + Crtn[base + jump]) * 32769) % F
                Crtn[base + jump] = ((Crtn[base] - Crtn[base + jump]) * var[(-i * block_num) % 16]) % F
    bp_done = time.time()
    print("Time spent decoding is: %f" % (bp_done - bp_received))

    # Verify correctness
    # Bit reverse the order to match the FFT
    # To obtain outputs in an ordinary order, bit reverse the order of input matrices prior to FFT
    bit_reverse = [0, 2, 1, 3]
    # Cver = [(Ap[bit_reverse[int(i / 4)]] * Bp[bit_reverse[i % 4]].T) % F for i in range(m * n)]
    # print([np.array_equal(Crtn[i], Cver[i]) for i in range(m * n)])

    row = Crtn[0]
    for j in range(1, 4):
        row = np.concatenate((row, Crtn[bit_reverse[j]]), axis=1)
    Cres = row
    for i in range(1, 4):
        row = Crtn[4 * bit_reverse[i]]
        for j in range(1, 4):
            row = np.concatenate((row, Crtn[4 * bit_reverse[i] + bit_reverse[j]]), axis=1)
        Cres = np.concatenate((Cres, row), axis=0)

    # print('A:')
    # print(A)
    # print('B:')
    # print(B)
    print('c:')
    print(Cres)
    print('dot:')
    dot = np.dot(A, B.T)
    print(dot)
    # print('divided')
    # print(Cres / dot)
    print('equal?')
    print(Cres == np.dot(A, B.T) % F)


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
    Ai = np.empty_like(np.matrix([[0.0] * s for i in range(int(r / m))]))
    Bi = np.empty_like(np.matrix([[0.0] * s for i in range(int(t / n))]))

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

    Ci = (Ai * Bi.T) % F

    wbp_done = time.time()
    # print "Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received)

    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()
