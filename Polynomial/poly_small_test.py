import numpy as np
from mpi4py import MPI
import polynomial_code
import logging
import time
LARGE_PRIME_NUMBER = 2125991977  # 65537
NP_DATA_TYPE=np.float64
# A = s x r, B = s x t

# A = np.arange(2*3).reshape([2, 3])
# B = np.ones([2, 1])
# m = 3
# n = 1

# s, r, t = 4, 4, 2
# A = np.arange(1, 1+s*r).reshape([s, r]).astype(NP_DATA_TYPE)
# B = np.ones([s, t]).astype(NP_DATA_TYPE)
# m, n = 4, 1


s, r, t = 500, 3073, 10
A = np.arange(1, 1+s*r).reshape([s, r]).astype(NP_DATA_TYPE)
A = np.random.rand(s, r).astype(dtype=NP_DATA_TYPE)
B = np.ones([s, t]).astype(NP_DATA_TYPE)
m, n = 7, 1

# loaded_data = np.load("wx_array.npz")
# A = loaded_data['X'].T
# B = loaded_data['W']
# m, n = 1, 1

N = m * n + 1
if __name__ == '__main__':
    # print ("rank: ", MPI.COMM_WORLD.Get_rank())
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    if MPI.COMM_WORLD.Get_rank() == 0:
        logging.info("A:\n" + str(A))
        logging.info("B:\n" + str(B))
        p_code = polynomial_code.PolynomialCoder(A, B, m, n, None, LARGE_PRIME_NUMBER, N, MPI.COMM_WORLD)
        p_code.polynomial_code()
        start = time.time()
        res = np.matmul(A.T, B)
        end = time.time()
        logging.info("np.matmul: time= " + str((end - start)) + "\n" + str(res))
    else:
        polynomial_code.PolynomialCoder.mapper(MPI.COMM_WORLD, False, False)
