from __future__ import print_function
import numpy as np
from data_utils import load_CIFAR10
from mpi4py import MPI
import time
from linear_svm import svm_loss_vectorized
from linear_svm import svm_loss_distributed

cifar10_dir = 'Datasets/cifar-10-batches-py'
comm = MPI.COMM_WORLD



loaded = True
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

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


W = np.random.randn(3073, 10) * 0.0001


tic = time.time()
loss_vectorized, grad_naive = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))


tic = time.time()
loss_distributed, _ = svm_loss_distributed(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Distributed loss: %e computed in %fs' % (loss_distributed, toc - tic))

print('difference: %f' % (loss_vectorized - loss_distributed))
