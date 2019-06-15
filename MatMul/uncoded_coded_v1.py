# !/usr/bin/env python
"""
Uncoded matrix multiplication
"""

from mpi4py import MPI
import numpy as np
import random
import threading
import time

# Change to True for more accurate timing, sacrificing performance
barrier = True
# Change to True to imitate straggler effects
straggling = True
timing = False


def loop():
    t = time.time()
    while time.time() < t + 6:
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

# Field size assumed to be prime for this implementation
F = 65537

# Input matrix size - A: s by r, B: s by t
s = 3000   # 3000   # 400
r = 100    # 100    # 500
t = 200    # 200    # 100

# Values of x_i used by 17 workers
var = [np.exp(2 * np.pi * i * 1j / 16) for i in range(16)] + [1]
#########################################################

comm = MPI.COMM_WORLD
A = np.zeros((r, s))
B = np.zeros((t, s))
if comm.rank == 0:
    A = np.matrix(np.random.randn(r, s))
    B = np.matrix(np.random.randn(t, s))

NUM_SIM_UNCODED = 100
MEAN_TIME_UNCODED = 0
START_UNCODED = 0
END_UNCODED = 0

if comm.rank == 0:
    print("UNCODED:")
for num_sim in range(NUM_SIM_UNCODED):
    if comm.rank == 0:
        # Master
        START_UNCODED = time.time()
        if timing:
            print("Running with %d processes:" % comm.Get_size())

        # Decide and broadcast chosen straggler
        # straggler = random.randint(1, N + 1)
        straggler = 1
        for i in range(N):
            comm.send(straggler, dest=i + 1, tag=7)

        # Split the matrices
        Ap = np.split(A, m)
        Bp = np.split(B, n)

        # Initialize return dictionary
        Crtn = []
        buf = np.zeros((int(r / m), int(t / n)), dtype=np.float)
        for i in range(N):
            Crtn.append(np.zeros((int(r / m), int(t / n)), dtype=np.float))

        # Start requests to send and receive
        reqA = [None] * N
        reqB = [None] * N
        reqC = [None] * N

        bp_start = time.time()

        for i in range(N - 1):
            reqA[i] = comm.Isend(Ap[i % m], dest=i + 1, tag=15)
            reqB[i] = comm.Isend(Bp[int(i / m)], dest=i + 1, tag=29)
            reqC[i] = comm.Irecv(Crtn[i], source=i + 1, tag=42)

        reqA[N - 1] = comm.Isend(Ap[0], dest=N, tag=15)
        reqB[N - 1] = comm.Isend(Bp[0], dest=N, tag=29)
        reqC[N - 1] = comm.Irecv(buf, source=N, tag=42)

        MPI.Request.Waitall(reqA)
        MPI.Request.Waitall(reqB)

        # Optionally wait for all workers to receive their submatrices, for more accurate timing
        if barrier:
            comm.Barrier()

        bp_sent = time.time()
        if timing:
            print("Time spent sending all messages is: %f" % (bp_sent - bp_start))

        MPI.Request.Waitall(reqC)
        bp_received = time.time()
        if timing:
            print("Time spent waiting for all workers is: %f" % (bp_received - bp_sent))

        END_UNCODED = time.time()
        if timing:
            print("TIME:")
            print(END_UNCODED - START_UNCODED)

    else:
        straggler = comm.recv(source=0, tag=7)

        Ai = np.empty_like(np.matrix([[0.0] * s for i in range(int(r / m))]))
        Bi = np.empty_like(np.matrix([[0.0] * s for i in range(int(t / n))]))
        rA = comm.Irecv(Ai, source=0, tag=15)
        rB = comm.Irecv(Bi, source=0, tag=29)

        rA.wait()
        rB.wait()

        if barrier:
            comm.Barrier()
        wbp_received = time.time()

        if straggling:
            if straggler == comm.rank:
                loop()
                # thread = threading.Thread(target=loop)
                # thread.start()

        Ci = (Ai * Bi.T) % F
        wbp_done = time.time()
        if timing:
            print("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

        sC = comm.Isend(Ci, dest=0, tag=42)
        sC.Wait()
    if comm.rank == 0:
        MEAN_TIME_UNCODED = (MEAN_TIME_UNCODED * num_sim + END_UNCODED - START_UNCODED) / (num_sim + 1)

if comm.rank == 0:
    print(MEAN_TIME_UNCODED)

NUM_SIM_CODED = 100
MEAN_TIME_CODED = 0
MEAN_TIME_DECODING_CODED = 0
START_CODED = 0
END_CODED = 0
START_DECODING_CODED = 0
END_DECODING_CODED = 0

if comm.rank == 0:
    print("CODED:")
for num_sim in range(NUM_SIM_CODED):
    if comm.rank == 0:
        # Master
        START_CODED = time.time()
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

        START_DECODING_CODED = time.time()
        if timing:
            print("Time spent waiting for %d workers %s is: %f" % (
                m * n, ",".join(map(str, [x + 1 for x in lst])), (START_DECODING_CODED - bp_sent)))

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

        END_DECODING_CODED = time.time()
        if timing:
            print("Time spent decoding is: %f" % (END_DECODING_CODED - START_DECODING_CODED))

        END_CODED = time.time()
        if timing:
            print("TIME:")
            print(END_CODED - START_CODED)


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
                pass
                # loop()
                # t = threading.Thread(target=loop)
                # t.start()

        # Ci = (Ai * Bi.T)
        Ci = np.dot(Ai, Bi.T)

        wbp_done = time.time()
        if timing:
            print("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

        sC = comm.Isend(Ci, dest=0, tag=42)
        sC.Wait()
    if comm.rank == 0:
        MEAN_TIME_CODED = (MEAN_TIME_CODED * num_sim + END_CODED - START_CODED) / (num_sim + 1)
        MEAN_TIME_DECODING_CODED = (MEAN_TIME_DECODING_CODED * num_sim + END_DECODING_CODED - START_DECODING_CODED) / (num_sim + 1)

if comm.rank == 0:
    print(MEAN_TIME_DECODING_CODED)
    print(MEAN_TIME_CODED)
