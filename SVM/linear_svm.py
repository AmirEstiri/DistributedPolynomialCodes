import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)

        correct_class_score = scores[y[i]]
        correct_class_coeff = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                correct_class_coeff -= 1
                dW[:, j] += X[i]

        dW[:, y[i]] += correct_class_coeff * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

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

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
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
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dScores = np.array(margins > 0, dtype=np.int32)
    num_nonzero = np.count_nonzero(dScores, axis=1)
    num_nonzero -= np.ones(num_nonzero.shape, dtype=np.int32)
    dScores[r, y] *= -num_nonzero

    dW = np.matmul(np.transpose(X), dScores)
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

from mpi4py import MPI
import sys
sys.path.insert(0,'../')
from Polynomial import polynomial_code
LARGE_PRIME_NUMBER = 2125991977
count = 0
def svm_distributed_loss_vectorized(W, X, y, reg):
    global count
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    scores = None

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    shape = (num_train, num_classes)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # scores = np.matmul(X, W)
    # print ("scores:", scores)
    NUMBER_OF_WORKERS = 6
    m = 5
    n = 1
    p_code = polynomial_code.PolynomialCoder(X.T, W, m, n, None, LARGE_PRIME_NUMBER, NUMBER_OF_WORKERS,
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
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dScores = np.array(margins > 0, dtype=np.int32)
    num_nonzero = np.count_nonzero(dScores, axis=1)
    num_nonzero -= np.ones(num_nonzero.shape, dtype=np.int32)
    dScores[r, y] *= -num_nonzero

    dW = np.matmul(np.transpose(X), dScores)
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
