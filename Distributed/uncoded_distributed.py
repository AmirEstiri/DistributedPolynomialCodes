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
    while time.time() < t + 2:
        a = 1 + 1


##################### Parameters ########################
# Use one master and N workers
N = 17

# Matrix division
m = 4
n = 4

# Input matrix size - A: s by r, B: s by t
s = 3073  # 3073  # 3
r = 500  # 500  # 8
t = 12  # 12  # 8

# CIFAR-10 constants

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Pick a primitive root 64
rt = 64

# Values of x_i used by 17 workers
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
    X = np.random.randn(r, s)
    W = np.random.randn(t, s)
    y = np.random.random_integers(0, 11, (r,))

START = time.time()
iter = 0
while acc < 0.4:
    iter += 1
    if comm.rank == 0:
        # Master
        START_ITER = time.time()

        cifar10_dir = 'Datasets/cifar-10-batches-py'

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

        for i in range(N-1):
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
        for k in range(m*n):
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
        print(np.real(loss))
        print(acc)

        END_ITER = time.time()
        print('iteration num:')
        print(iter)
        print('iteration time:')
        print(END_ITER - START_ITER)

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
                loop()
                # t = threading.Thread(target=loop)
                # t.start()

        Ci = (Ai * Bi.T)

        wbp_done = time.time()
        if timing:
            pass
            # print("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

        sC = comm.Isend(Ci, dest=0, tag=42)
        sC.Wait()

exit(0)
END = time.time()
print('total time:')
print(END - START)