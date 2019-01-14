# Run some setup code for this notebook.
from __future__ import print_function
import sys
import logging
import numpy as np
from cs231n.data_utils import load_CIFAR10
from mpi4py import MPI

sys.path.insert(0, '../')
from SVM import linear_svm
from SVM.linear_svm import svm_distributed_loss_vectorized
from Polynomial import polynomial_code

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

def master():
    # Load the raw CIFAR-10 data.
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the training data;
    # we can use this for development so our code runs faster.
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # We will also make a development set, which is a small subset of
    # the training set.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # As a sanity check, print out the shapes of the data
    print('Training data shape: ', X_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Test data shape: ', X_test.shape)
    print('dev data shape: ', X_dev.shape)

    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)

    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    # Compute the loss and its gradient at W.
    loss, grad = svm_distributed_loss_vectorized(W, X_dev, y_dev, 0.0)

    # Numerically compute the gradient along several randomly chosen dimensions, and
    # compare them with your analytically computed gradient. The numbers should match
    # almost exactly along all dimensions.
    from cs231n.gradient_check import grad_check_sparse
    f = lambda w: svm_distributed_loss_vectorized(w, X_dev, y_dev, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad)

    # do the gradient check once again with regularization turned on
    # you didn't forget the regularization gradient did you?
    loss, grad = svm_distributed_loss_vectorized(W, X_dev, y_dev, 5e1)
    f = lambda w: svm_distributed_loss_vectorized(w, X_dev, y_dev, 5e1)[0]
    grad_numerical = grad_check_sparse(f, W, grad)


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    if mpi_rank == 0:
        master()
    else:
        polynomial_code.PolynomialCoder.mapper(mpi_comm, False, False)

