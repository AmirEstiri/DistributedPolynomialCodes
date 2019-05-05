import random
from mpi4py import MPI
import numpy as np
import time
from data_utils import load_CIFAR10

comm = MPI.COMM_WORLD
cifar10_dir = 'Datasets/cifar-10-batches-py'
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500
X_train, y_train, X_test, y_test, X_dev, y_dev = np.zeros((1, 1)), np.zeros((1, 1)), \
                                                 np.zeros((1, 1)), np.zeros((1, 1)), \
                                                 np.zeros((1, 1)), np.zeros((1, 1))
W = np.random.randn(3073, 10) * 0.0001
if comm.Get_rank() == 0:
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


X, y, reg = X_dev, y_dev, 0.000005
scores = None
loss = 0.0
dW = np.zeros(W.shape)
num_classes = W.shape[1]
num_train = X.shape[0]
shape = (num_train, num_classes)
F = 65537
N = 17
m = 4
n = 4
s = X.T.shape[0]
r = X.T.shape[1]
t = W.shape[1]
NP_DATA_TYPE = np.float64
MPI_DATA_TYPE = MPI.DOUBLE
var = [pow(64, i, F) for i in range(16)] + [3]


if not r % m == 0:
    new_r = r + m - r % m
    X = np.pad(X, ((0, 0), (0, new_r - r)), 'constant')
    r = new_r
if not s % n == 0:
    new_t = t + n - t % n
    W = np.pad(W, ((0, 0), (0, new_t - t)), 'constant')
    t = new_t

if comm.Get_rank() == 0:
    spec_dict = [r, t, s, F, n, m, N]
    comm.bcast(spec_dict)

    straggler = random.randint(1, N)
    for i in range(N):
        comm.send(straggler, dest=i + 1, tag=7)

    Ap = np.hsplit(X.T, m)
    Bp = np.hsplit(W, n)

    # Encode the matrices
    Aenc = [sum([Ap[j] * (pow(var[i], j, F)) for j in range(m)]) % F for i in range(N)]
    Benc = [sum([Bp[j] * (pow(var[i], j * m, F)) for j in range(n)]) % F for i in range(N)]

    request_A = [None] * N
    request_B = [None] * N

    for i in range(N):
        request_A[i] = comm.Isend([Aenc[i], MPI_DATA_TYPE], dest=i + 1, tag=15)
        if i == 10:
            print('sent:')
            Aenc = np.array(Aenc)
            print(Aenc.shape)
        request_B[i] = comm.Isend([Benc[i], MPI_DATA_TYPE], dest=i + 1, tag=29)

    MPI.Request.Waitall(request_A)
    MPI.Request.Waitall(request_B)

    ###################
    return_dict = []
    for i in range(N):
        return_dict.append(np.zeros((int(r / m), int(t / n)), dtype=NP_DATA_TYPE))

    # Start requests to receive
    request_C = [None] * N
    for i in range(N):
        request_C[i] = comm.Irecv([return_dict[i], MPI_DATA_TYPE], source=i + 1, tag=42)

    return_C = [None] * N
    recv_index = []

    for i in range(m * n):
        j = MPI.Request.Waitany(request_C)
        recv_index.append(j)
        return_C[j] = return_dict[j]

    missing = set(range(m * n)) - set(recv_index)

    for i in missing:
        begin = time.time()
        coeff = [1] * (m * n)
        for j in range(m * n):
            # Compute coefficient
            for k in set(recv_index) - set([recv_index[j]]):
                coeff[j] = (coeff[j] * (var[i] - var[k]) * pow(var[recv_index[j]] - var[k], F - 2, F)) % F
        return_C[i] = sum([return_C[recv_index[j]] * coeff[j] for j in range(16)]) % F

    for k in range(4):
        jump = 2 ** (3 - k)
        for i in range(jump):
            block_num = int(8 / jump)
            for j in range(block_num):
                base = i + j * jump * 2
                return_C[base] = ((return_C[base] + return_C[base + jump]) * 32769) % F
                return_C[base + jump] = ((return_C[base] - return_C[base + jump]) * var[(-i * block_num) % 16]) % F
    print(return_C)
    res = np.matmul(X, W)
    print(res)
else:
    spec_dict = comm.bcast(None)
    # r, t, s, F, n, m, N = spec_dict

    # Receive straggler information from the master
    straggler = comm.recv(source=0, tag=7)

    # Receive split input matrices from the master
    Ai = np.empty_like(np.matrix([[0] * int(r / m) for i in range(s)]), dtype=NP_DATA_TYPE)
    Bi = np.empty_like(np.matrix([[0] * int(t / n) for i in range(s)]), dtype=NP_DATA_TYPE)
    receive_A = comm.Irecv(Ai, source=0, tag=15)
    if comm.Get_rank() == 10:
        print('received:')
        print(Ai.shape)
    receive_B = comm.Irecv(Bi, source=0, tag=29)

    receive_A.wait()
    receive_B.wait()

    Ci = (Ai.T * Bi) % F

    sC = comm.Isend([Ci, MPI_DATA_TYPE], dest=0, tag=42)
    sC.Wait()

# scores = p_code.coeffs
# r = range(num_train)
# correct_scores_mat = np.repeat(scores[r, y], num_classes).reshape(num_train, num_classes)
#
# mask = np.ones(shape, dtype=bool)
# mask[r, y] = False
#
# margins = np.maximum(np.zeros(shape),
#                      scores - correct_scores_mat + np.ones(shape))
#
# loss = np.sum(margins[mask])  # only incorrect classes are considered in computing loss.
# loss /= num_train
# loss += reg * np.sum(W * W)
#
# dScores = np.array(margins > 0, dtype=np.int32)
# num_nonzero = np.count_nonzero(dScores, axis=1)
# num_nonzero -= np.ones(num_nonzero.shape, dtype=np.int32)
# dScores[r, y] *= -num_nonzero
#
# dW = np.matmul(np.transpose(X), dScores)
# dW = dW / num_train + 2 * reg * W
