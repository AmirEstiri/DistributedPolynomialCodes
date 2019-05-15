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
s = 3#3073#3
r = 8#500#8
t = 8#12#8

# Pick a primitive root 64
rt = 64

# Values of x_i used by 17 workers
var = [pow(64, i, 65537) for i in range(16)] + [3]
#########################################################

comm = MPI.COMM_WORLD

if comm.rank == 0:
    # Master
    print("Running with %d processes:" % comm.Get_size())

    # Decide and broadcast chosen straggler

    straggler = random.randint(1, N)
    for i in range(N):
        comm.send(straggler, dest=i + 1, tag=7)

    # Create random matrices of 8-bit ints
    A = np.matrix(np.random.random_integers(0, 255, (r, s)))
    B = np.matrix(np.random.random_integers(0, 255, (t, s)))

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

    print(Aenc[0])

    for i in range(N):
        reqA[i] = comm.Isend([Aenc[i], MPI.DOUBLE], dest=i + 1, tag=15)
        reqB[i] = comm.Isend([Benc[i], MPI.DOUBLE], dest=i + 1, tag=29)
        reqC[i] = comm.Irecv([Rdict[i], MPI.DOUBLE], source=i + 1, tag=42)

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
    # print ([np.array_equal(Crtn[i], Cver[i]) for i in range(m * n)])

    row = Crtn[0]
    for j in range(1, 4):
        row = np.concatenate((row, Crtn[bit_reverse[j]]), axis=1)
    Cres = row
    for i in range(1, 4):
        row = Crtn[4 * bit_reverse[i]]
        for j in range(1, 4):
            row = np.concatenate((row, Crtn[4 * bit_reverse[i] + bit_reverse[j]]), axis=1)
        print(Cres.shape)
        print(row.shape)
        Cres = np.concatenate((Cres, row), axis=0)

    print('c:')
    print(Cres)
    print('dot:')
    print(np.dot(A, B.T))

    print('equal?')
    print(Cres == np.dot(A, B.T) % F)


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

    Ci = (Ai * Bi.T) % F
    wbp_done = time.time()
    # print "Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received)

    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()
