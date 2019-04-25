import numpy as np
from mpi4py import MPI
from polynomial_code import PolynomialCoder
import logging
import time

# LARGE_PRIME_NUMBER = 2125991977  # 65537
LARGE_PRIME_NUMBER = 65537
NP_DATA_TYPE = np.float64
# A = s x r, B = s x t

# A = np.arange(2*3).reshape([2, 3])
# B = np.ones([2, 1])
# m = 3
# n = 1

# s, r, t = 4, 4, 2
# A = np.arange(1, 1+s*r).reshape([s, r]).astype(NP_DATA_TYPE)
# B = np.ones([s, t]).astype(NP_DATA_TYPE)
# m, n = 4, 1


## s, r, t = 500, 3073, 10
## A = np.arange(1, 1 + s * r).reshape([s, r]).astype(NP_DATA_TYPE)
## A = np.random.rand(s, r).astype(dtype=NP_DATA_TYPE)
## B = np.ones([s, t]).astype(NP_DATA_TYPE)
## m, n = 7, 1

s, r, t, m, n = 5, 8, 8, 4, 4
A = np.matrix([[1, 2, 1, 0, 1, 2, 3, 1], [0, 0, 1, 1, 0, 0, 4, 4], [1, 5, 4, 4, 5, 1, 0, 1], [2, 2, 3, 4, 2, 1, 0, 2],
               [6, 6, 7, 7, 8, 1, 0, 9]])

B = np.matrix([[1, 0, 1, 1, 1, 0, 9, 8], [7, 7, 6, 9, 8, 5, 6, 9], [6, 8, 1, 4, 3, 2, 5, 9], [7, 6, 7, 9, 1, 0, 2, 5],
               [5, 4, 8, 9, 1, 7, 6, 2]])

# loaded_data = np.load("wx_array.npz")
# A = loaded_data['X'].T
# B = loaded_data['W']
# m, n = 1, 1

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
        # print res == p_code.buffer
    else:
        p_code.mapper()
