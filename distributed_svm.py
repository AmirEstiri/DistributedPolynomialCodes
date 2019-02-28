from Polynomial.polynomial_code import PolynomialCoder
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
if comm.rank == 0:
    X = np.matrix(np.random.random_integers(0, 255, (4000, 100)))
    y = np.matrix(np.random.random_integers(0, 255, (4000, 1)))
    W = np.zeros((100, 1))

    num_iters = 100
    batch_size = 200
    verbose = False
    reg = 1e-5
    learning_rate = 1e-3
    # coder = PolynomialCoder(X, y, 4, 4, W, 17, comm)
    num_train, dim = X.shape
    num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
    if W is None:
        # lazily initialize W
        W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        X_batch = None
        y_batch = None

        ind = np.random.choice(num_train, batch_size)
        X_batch = X[ind]
        y_batch = y[ind]

        # evaluate loss and gradient

        # START

        loss = 0.0
        dW = np.zeros(W.shape)  # initialize the gradient as zero
        num_classes = W.shape[1]
        num_train = X_batch.shape[0]
        shape = (num_train, num_classes)

        scores = np.matmul(X_batch, W)
        coder = PolynomialCoder()
        r = range(num_train)
        # matrix of size num_train * num_classes. All of the elemements of the i-th
        # row is the score of correct class of i-th instance.
        correct_scores_mat = np.repeat(scores[r, y_batch], num_classes).reshape(num_train, num_classes)

        mask = np.ones(shape, dtype=bool)
        mask[r, y_batch] = False

        margins = np.maximum(np.zeros(shape),
                             scores - correct_scores_mat + np.ones(shape))

        loss = np.sum(margins[mask])  # only incorrect classes are considered in computing loss.
        loss /= num_train
        loss += reg * np.sum(W * W)
        dScores = np.array(margins > 0, dtype=np.int32)
        num_nonzero = np.count_nonzero(dScores, axis=1)
        num_nonzero -= np.ones(num_nonzero.shape, dtype=np.int32)
        dScores[r, y_batch] *= -num_nonzero

        dW = np.matmul(np.transpose(X_batch), dScores)
        dW = dW / num_train + 2 * reg * W

        # END
        grad = dW

        loss_history.append(loss)

        # perform parameter update
        W -= learning_rate * grad

        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))
else:


