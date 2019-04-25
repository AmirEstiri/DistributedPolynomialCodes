import numpy as np


from mpi4py import MPI
import sys
from .polynomial_code import PolynomialCoder

sys.path.insert(0, '../')

LARGE_PRIME_NUMBER = 65537
count = 0


def svm_distributed_loss_vectorized(W, X, y, reg):
    global count
    """
    Structured SVM loss function, naive implementation (with loops).
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    scores = None

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    shape = (num_train, num_classes)
    NUMBER_OF_WORKERS = 6
    m = 5
    n = 1
    p_code = PolynomialCoder(X.T, W, m, n, None, LARGE_PRIME_NUMBER, NUMBER_OF_WORKERS,
                                             MPI.COMM_WORLD)
    p_code.polynomial_code()
    scores = p_code.coeffs
    r = range(num_train)
    # matrix of size num_train * num_classes. All of the elemements of the i-th
    # row is the score of correct class of i-th instance.
    correct_scores_mat = np.repeat(scores[r, y], num_classes).reshape(num_train, num_classes)

    mask = np.ones(shape, dtype=bool)
    mask[r, y] = False

    margins = np.maximum(np.zeros(shape),
                         scores - correct_scores_mat + np.ones(shape))

    loss = np.sum(margins[mask])  # only incorrect classes are considered in computing loss.
    loss /= num_train
    loss += reg * np.sum(W * W)

    dScores = np.array(margins > 0, dtype=np.int32)
    num_nonzero = np.count_nonzero(dScores, axis=1)
    num_nonzero -= np.ones(num_nonzero.shape, dtype=np.int32)
    dScores[r, y] *= -num_nonzero

    dW = np.matmul(np.transpose(X), dScores)
    dW = dW / num_train + 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    shape = (num_train, num_classes)

    scores = np.matmul(X, W)
    r = range(num_train)
    # matrix of size num_train * num_classes. All of the elemements of the i-th
    # row is the score of correct class of i-th instance.
    correct_scores_mat = np.repeat(scores[r, y], num_classes).reshape(num_train, num_classes)

    mask = np.ones(shape, dtype=bool)
    mask[r, y] = False

    margins = np.maximum(np.zeros(shape),
                         scores - correct_scores_mat + np.ones(shape))

    loss = np.sum(margins[mask])  # only incorrect classes are considered in computing loss.
    loss /= num_train
    loss += reg * np.sum(W * W)
    dScores = np.array(margins > 0, dtype=np.int32)
    num_nonzero = np.count_nonzero(dScores, axis=1)
    num_nonzero -= np.ones(num_nonzero.shape, dtype=np.int32)
    dScores[r, y] *= -num_nonzero

    dW = np.matmul(np.transpose(X), dScores)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
