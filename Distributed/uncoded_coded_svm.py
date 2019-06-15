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
straggling = True
timing = False


def loop():
    t = time.time()
    while time.time() < t + 3:
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
s = 3073
r = 500
t = 12

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
    X = np.random.randn(r, s)
    W = np.random.randn(t, s)
    y = np.random.random_integers(0, t - 1, (r,))

if comm.rank == 0:
    print("CODED:")
MEAN_TIME_CODED = 0
START_CODED = 0
MEAN_TIME_ITER_CODED = 0
MEAN_NUM_ITER_CODED = 0
NUM_SIM_CODED = 5
START_ITER_CODED = 0
iter = 0
ctr = 0

for num_sim in range(NUM_SIM_CODED):
    START_CODED = time.time()
    ctr = 0
    W = np.random.randn(t, s)
    while acc < 0.4:
        if comm.rank == 0:
            START_ITER_CODED = time.time()
            # Master
            if timing:
                print("Running with %d processes:" % comm.Get_size())

            # Decide and broadcast chosen straggler

            straggler = random.randint(1, N)
            for i in range(N):
                comm.send(straggler, dest=i + 1, tag=7)

            A = X
            B = W

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

            # Calculate SVM loss
            num_classes = t
            num_train = r
            shape = (r, t)
            #
            scores = Cres
            # matrix of size num_train * num_classes. All of the elemements of the i-th
            # row is the score of correct class of i-th instance.
            correct_scores_mat = np.repeat(scores[range(num_train), y], num_classes).reshape(num_train, num_classes)

            mask = np.ones(shape, dtype=bool)
            mask[range(num_train), y] = False

            margins = np.maximum(np.zeros(shape),
                                 scores - correct_scores_mat + np.ones(shape))

            loss = np.sum(margins[mask])  # only incorrect classes are considered in computing loss.
            loss /= num_train
            loss += reg * np.sum(W * W)
            dScores = np.array(margins > 0, dtype=np.int32)
            num_nonzero = np.count_nonzero(dScores, axis=1)
            num_nonzero -= np.ones((r,), dtype=np.int32)
            dScores[range(num_train), y] *= -num_nonzero

            dW = np.dot(np.transpose(X), dScores).T
            dW = dW / num_train + 2 * reg * W

            W -= dW * learning_rate
            y_pred = np.argmax(scores, axis=1)
            acc = np.sum(y_pred == y) / y.shape[0]
            # End SVM loss


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

            Ci = (Ai * Bi.T)

            wbp_done = time.time()
            if timing:
                pass
                # print("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

            sC = comm.Isend(Ci, dest=0, tag=42)
            sC.Wait()

        if comm.rank == 0:
            if True:
                print(np.real(loss))
                print(acc)

            END_ITER_CODED = time.time()
            MEAN_TIME_ITER_CODED = (MEAN_TIME_ITER_CODED * iter + END_ITER_CODED - START_ITER_CODED) / (iter + 1)
            if True:
                print('iteration num: %d' % iter)
                print('iteration time: %f' % (END_ITER_CODED - START_ITER_CODED))
            iter += 1
            ctr += 1

    END_CODED = time.time()
    MEAN_TIME_CODED = (MEAN_TIME_CODED * num_sim + END_CODED - START_CODED) / (num_sim + 1)
    MEAN_NUM_ITER_CODED = (MEAN_NUM_ITER_CODED * num_sim + ctr) / (num_sim + 1)

if comm.rank == 0:
    print(MEAN_TIME_ITER_CODED)
    print(MEAN_NUM_ITER_CODED)
    print(MEAN_TIME_CODED)


if comm.rank == 0:
    print("UNCODED:")
MEAN_TIME_UNCODED = 0
START_UNCODED = 0
MEAN_TIME_ITER_UNCODED = 0
MEAN_NUM_ITER_UNCODED = 0
NUM_SIM_UNCODED = 5
START_ITER_UNCODED = 0
iter = 0
ctr = 0
num_sim = 0

for num_sim in range(NUM_SIM_UNCODED):
    START_UNCODED = time.time()
    ctr = 0
    acc = 0
    W = np.random.randn(t, s)
    while acc < 0.4:
        if comm.rank == 0:
            # Master
            START_ITER_UNCODED = time.time()

            if timing:
                print("Running with %d processes:" % comm.Get_size())

            # Decide and broadcast chosen straggler

            straggler = random.randint(1, N)
            for i in range(N):
                comm.send(straggler, dest=i + 1, tag=7)

            A = X
            B = W

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
                reqB[i] = comm.Isend(Bp[int((i - i % m) / m)], dest=i + 1, tag=29)
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

            # TODO:
            Cres = np.zeros((r, t), dtype=float)
            jump_x = int(r / 4)
            jump_y = int(t / 4)
            for k in range(m * n):
                for i in range(jump_x):
                    for j in range(jump_y):
                        X_0 = jump_x * (k % 4)
                        Y_0 = jump_y * int((k - k % 4) / 4)
                        Cres[X_0 + i][Y_0 + j] = Crtn[k][i, j]

            bp_done = time.time()
            if timing:
                print("Time spent decoding is: %f" % (bp_done - bp_received))

            # Calculate SVM loss
            num_classes = t
            num_train = r
            shape = (r, t)

            scores = Cres
            # matrix of size num_train * num_classes. All of the elemements of the i-th
            # row is the score of correct class of i-th instance.
            correct_scores_mat = np.repeat(scores[range(num_train), y], num_classes).reshape(num_train, num_classes)

            mask = np.ones(shape, dtype=bool)
            mask[range(num_train), y] = False

            margins = np.maximum(np.zeros(shape),
                                 scores - correct_scores_mat + np.ones(shape))

            loss = np.sum(margins[mask])  # only incorrect classes are considered in computing loss.
            loss /= num_train
            loss += reg * np.sum(W * W)
            dScores = np.array(margins > 0, dtype=np.int32)
            num_nonzero = np.count_nonzero(dScores, axis=1)
            num_nonzero -= np.ones(num_nonzero.shape, dtype=np.int32)
            dScores[range(num_train), y] *= -num_nonzero

            dW = np.matmul(np.transpose(X), dScores).T
            dW = dW / num_train + 2 * reg * W
            W -= dW * learning_rate

            y_pred = np.argmax(scores, axis=1)
            acc = np.sum(y_pred == y) / y.shape[0]

            # End SVM loss


        else:
            # Worker
            # Receive straggler information from the master
            straggler = comm.recv(source=0, tag=7)

            # Receive split input matrices from the master
            Ai = np.empty_like(np.matrix([[0.0] * s for i in range(int(r / m))]))
            Bi = np.empty_like(np.matrix([[0.0] * s for i in range(int(t / n))]))

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
                    # pass
                    loop()

            Ci = (Ai * Bi.T)

            wbp_done = time.time()
            if timing:
                pass
                # print("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

            sC = comm.Isend(Ci, dest=0, tag=42)
            sC.Wait()

        if comm.rank == 0:
            if timing:
                print(np.real(loss))
                print(acc)

            END_ITER_UNCODED = time.time()
            MEAN_TIME_ITER_UNCODED = (MEAN_TIME_ITER_UNCODED * iter + END_ITER_UNCODED - START_ITER_UNCODED) / (iter + 1)
            if timing:
                print('iteration num: %d' % iter)
                print('iteration time: %f' % (END_ITER_UNCODED - START_ITER_UNCODED))
            iter += 1
            ctr += 1

    END_UNCODED = time.time()
    MEAN_TIME_UNCODED = (MEAN_TIME_UNCODED * num_sim + END_UNCODED - START_UNCODED) / (num_sim + 1)
    MEAN_NUM_ITER_UNCODED = (MEAN_NUM_ITER_UNCODED * num_sim + ctr) / (num_sim + 1)

if comm.rank == 0:
    print(MEAN_TIME_ITER_UNCODED)
    print(MEAN_NUM_ITER_UNCODED)
    print(MEAN_TIME_UNCODED)
