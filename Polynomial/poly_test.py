#!/usr/bin/env python
"""
Polynomial code with fast decoding
"""

from mpi4py import MPI
import numpy as np
import random
import threading
import time
from scipy.interpolate import lagrange

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

# Field size assumed to be prime for this implementation
F = 2125991977

s, r, t, m, n = 5, 8, 8, 4, 4
A = np.matrix([[1, 2, 1, 0, 1, 2, 3, 1], [0, 0, 1, 1, 0, 0, 4, 4], [1, 5, 4, 4, 5, 1, 0, 1], [2, 2, 3, 4, 2, 1, 0, 2],
               [6, 6, 7, 7, 8, 1, 0, 9]])

B = np.matrix([[1, 0, 1, 1, 1, 0, 9, 8], [7, 7, 6, 9, 8, 5, 6, 9], [6, 8, 1, 4, 3, 2, 5, 9], [7, 6, 7, 9, 1, 0, 2, 5],
               [5, 4, 8, 9, 1, 7, 6, 2]])

# Pick a primitive root 64
rt = 64

# Values of x_i used by 17 workers
var = [i + 1 for i in range(17)]

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
    # A = np.matrix(np.random.random_integers(0, 255, (r, s)))
    # B = np.matrix(np.random.random_integers(0, 255, (t, s)))
    A = np.matrix([[1, 2, 0, 1], [1, 3, 1, 1], [3, 3, 4, 1], [1, 0, 2, 1]])
    B = np.matrix([[1, 3], [2, 0], [3, 0], [1, 0]])

    # Split the matrices
    Ap = np.hsplit(A, m)
    Bp = np.hsplit(B, n)

    # Encode the matrices
    Aenc = [sum([Ap[j] * (pow(var[i], j, F)) for j in range(m)]) % F for i in range(N)]
    Benc = [sum([Bp[j] * (pow(var[i], j * m, F)) for j in range(n)]) % F for i in range(N)]


    # Initialize return dictionary
    Rdict = []
    for i in range(N):
        Rdict.append(np.zeros((int(r / m), int(t / n)), dtype=np.int_))

    # Start requests to send and receive
    # reqA = [None] * N
    # reqB = [None] * N
    # reqC = [None] * N
    reqA = []
    reqB = []
    reqC = []

    bp_start = time.time()

    for i in range(N):
        reqA.append(comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15))
        reqB.append(comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29))
        req = MPI.Request
        reqC.append(comm.Irecv([Rdict[i], MPI.INT], source=i + 1, tag=42))

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
        for i in range(jump):
            block_num = int(8 / jump)
            for j in range(block_num):
                base = i + j * jump * 2
                Crtn[base] = ((Crtn[base] + Crtn[base + jump]) * 32769) % F
                Crtn[base + jump] = ((Crtn[base] - Crtn[base + jump]) * var[(-i * block_num) % 16]) % F

    # coeffs = np.zeros((int(r / m), int(t / n), m * n))
    # for i in range(int(r / m)):
    #     for j in range(int(t / n)):
    #         f_z = []
    #         for k in range(m * n):
    #             f_z.append(Crtn[lst[k]][i][j])
    #         lagrange_interpolate = lagrange(lst, f_z)
    #         coeffs[i][j] = lagrange_interpolate
    # print(coeffs)

    bp_done = time.time()
    print("Time spent decoding is: %f" % (bp_done - bp_received))

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
    # Ai = np.empty_like(np.matrix([[0] * s for i in range(int(r / m))]))
    Ai = np.empty_like(np.matrix([[0] * int(r / m) for i in range(s)]))
    # Ai = Ai.getT()
    Bi = np.empty_like(np.matrix([[0] * int(t / n) for i in range(s)]))
    # Bi = Bi.getT()

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
    Ci = (Ai.getT() * Bi) % F
    wbp_done = time.time()
    # print("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()
