from data_handler import read_data_cancer
from Polynomial.polynomial_code import PolynomialCoder
import numpy as np
from mpi4py import MPI


X, y = read_data_cancer()
N = X.shape[0]
D = X.shape[1]
C = y.shape[1]
W = np.zeros((D, C))
X0 = np.zeros((N, D))
y0 = np.zeros((N, C))
W0 = np.zeros((D, C))
m = 4
n = 4
F = 65537
comm = MPI.COMM_WORLD
NODES = 17
poly_coder = PolynomialCoder(X0, y0, m, n, W0, F, NODES, comm)
