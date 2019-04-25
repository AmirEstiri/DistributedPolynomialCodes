import numpy as np
from mpi4py import MPI
from Distributed.polynomial_code import PolynomialCoder
import logging
import time

# LARGE_PRIME_NUMBER = 2125991977
LARGE_PRIME_NUMBER = 65537
NP_DATA_TYPE = np.float64


s, r, t, m, n = 5, 8, 8, 4, 4
A = np.matrix(np.random.random_integers(0, 255, (r, s)))
B = np.matrix(np.random.random_integers(0, 255, (t, s)))
A_prime = np.ndarray(shape=(r, s), dtype=int)
A_prime = np.array(
    [[1, 2, 1, 0, 1, 2, 3, 1], [0, 0, 1, 1, 0, 0, 4, 4], [1, 5, 4, 4, 5, 1, 0, 1], [2, 2, 3, 4, 2, 1, 0, 2],
     [6, 6, 7, 7, 8, 1, 0, 9]])
A_prime = np.matrix(A_prime).T

B_prime = np.ndarray(shape=(t, s), dtype=int)
B_prime = np.array(
    [[1, 0, 1, 1, 1, 0, 9, 8], [7, 7, 6, 9, 8, 5, 6, 9], [6, 8, 1, 4, 3, 2, 5, 9], [7, 6, 7, 9, 1, 0, 2, 5],
     [5, 4, 8, 9, 1, 7, 6, 2]])
B_prime = np.matrix(B_prime).T
#
for i in range(r):
    for j in range(s):
        A[i, j] = A_prime[i, j]
        B[i, j] = B_prime[i, j]
A = A.T
B = B.T

N = m * n + 1
if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    p_code = PolynomialCoder(A, B, m, n, None, LARGE_PRIME_NUMBER, N, MPI.COMM_WORLD)
    if MPI.COMM_WORLD.Get_rank() == 0:
        # logging.info("A:\n" + str(A))
        # logging.info("B:\n" + str(B))
        p_code.data_send()
        p_code.reducer()
        start = time.time()
        res = np.matmul(A.T, B)
        end = time.time()
        logging.info("np.matmul: time= " + str((end - start)) + "\n" + str(res))
    else:
        p_code.mapper()
