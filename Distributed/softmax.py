import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        scores = np.matmul(X[i], W)
        scores -= np.max(scores)
        p = np.exp(scores) / np.sum(np.exp(scores))
        loss += -np.log(p[y[i]])

        dScores = p
        dScores[y[i]] -= 1

        for j in range(num_class):
            dW[:, j] += dScores[j] * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]
    r = range(num_train)

    scores = np.matmul(X, W)
    maxs = np.max(scores, axis=1).repeat(num_class).reshape(num_train, num_class)
    scores -= maxs

    denoms = np.sum(np.exp(scores), axis=1).repeat(num_class).reshape(num_train, num_class)
    p = np.exp(scores) / denoms

    loss = -np.sum(np.log(p[r, y]))

    dScores = p
    dScores[r, y] -= 1
    dW = np.matmul(X.T, dScores)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################


    return loss, dW
