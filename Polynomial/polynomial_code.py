from mpi4py import MPI
import numpy as np
import random
import threading
import time
from scipy import interpolate
import logging

NP_DATA_TYPE = np.float64
MPI_DATA_TYPE = MPI.DOUBLE

def loop():
    t = time.time()
    while time.time() < t + 60:
        a = 1 + 1


class PolynomialCoder:

    def __init__(self, A, B, m, n, buffer, F, N, comm):
        # Buffer to put the answer
        self.buffer = buffer
        # Change to True for more accurate timing, sacrificing performance
        self.barrier = False
        # Change to True to imitate straggler effects
        self.straggling = False
        self.comm = comm
        self.N = N
        self.A = A.astype(NP_DATA_TYPE)
        self.B = B.astype(NP_DATA_TYPE)
        self.s = A.shape[0]
        self.r = A.shape[1]
        self.t = B.shape[1]
        self.m = m
        self.n = n
        self.var = [i+1 for i in range(N+1)] + [3]
        logging.debug("var:\n" + str(self.var))
        # self.zero_padding_matrices()
        self.F = F
        self.coeffs = None

    def zero_padding_matrices(self):
        s = self.s
        r = self.r
        m = self.m
        n = self.n
        A = self.A
        B = self.B
        t = self.t
        new_r = r + m - r % m
        A_pad = np.zeros((s, new_r))
        A_pad[:, :s] = A
        new_t = t + n - t % n
        B_pad = np.zeros((s, new_t))
        B_pad[:, :s] = B
        self.A = A_pad
        self.B = B_pad
        self.r = new_r
        self.t = new_t

    def data_send(self):
        comm = self.comm
        A = self.A
        B = self.B
        var = self.var
        F = self.F
        n = self.n
        m = self.m
        N = self.N

        spec_dict = [self.r, self.t, self.s, F, n, m, N]
        # logging.debug("r,t,s,F,n,m,N\n" + str(spec_dict))
        comm.bcast(spec_dict)

        # Decide and broadcast chosen straggler
        straggler = random.randint(1, N)
        for i in range(N):
            comm.send(straggler, dest=i + 1, tag=7)

        # Split matrices
        Ap = np.hsplit(A, m)
        Bp = np.hsplit(B, n)

        # Encode the matrices
        Aenc = [sum([Ap[j] * (pow(var[i], j, F)) for j in range(m)]) % F for i in range(N)]
        Benc = [sum([Bp[j] * (pow(var[i], j * m, F)) for j in range(n)]) % F for i in range(N)]

        logging.debug("Aenc:" + str(Aenc[0].dtype) + "\n" + str(Aenc))
        logging.debug("Benc:" + str(Benc[0].dtype) + "\n" + str(Benc))

        # Start requests to send
        request_A = [None] * N
        request_B = [None] * N
        self.bp_start = time.time()
        logging.debug("A[i]" + str(Aenc[i].shape) + ", dtype=" + str(Aenc[i].dtype))
        logging.debug("B[i]" + str(Benc[i].shape) + ", dtype=" + str(Benc[i].dtype))
        for i in range(N):
            request_A[i] = comm.Isend([Aenc[i], MPI_DATA_TYPE], dest=i + 1, tag=15)
            request_B[i] = comm.Isend([Benc[i], MPI_DATA_TYPE], dest=i + 1, tag=29)
        MPI.Request.Waitall(request_A)
        MPI.Request.Waitall(request_B)

        # Optionally wait for all workers to receive their submatrices, for more accurate timing
        if self.barrier:
            comm.Barrier()

        self.bp_sent = time.time()
        logging.info("Time spent sending all messages is: %f" % (self.bp_sent - self.bp_start))

    def reducer(self):

        comm = self.comm
        var = self.var
        F = self.F
        r = self.r
        n = self.n
        m = self.m
        t = self.t
        N = self.N

        # Initialize return dictionary
        return_dict = []
        for i in range(N):
            return_dict.append(np.zeros((int(r / m), int(t / n)), dtype=NP_DATA_TYPE))

        # Start requests to receive
        request_C = [None] * N
        for i in range(N):
            request_C[i] = comm.Irecv([return_dict[i], MPI_DATA_TYPE], source=i + 1, tag=42)

        return_C = [None] * N
        recv_index = []
        # Wait for the mn fastest workers
        for i in range(m * n):
            j = MPI.Request.Waitany(request_C)
            recv_index.append(j)
            return_C[j] = return_dict[j]

        self.bp_received = time.time()
        logging.info("Time spent waiting for %d workers %s is: %f" % (
            m * n, ",".join(map(str, [x + 1 for x in recv_index])), (self.bp_received - self.bp_sent)))

        logging.debug("return C: " + str(return_C))

        self.coeffs = self.calculate_C(return_C, recv_index)
        self.bp_done = time.time()
        logging.info("Time spent decoding is: %f" % (self.bp_done - self.bp_received))

    def calculate_base_indices(self):
        r, s, t, m, n = self.r, self.s, self.t, self.m, self.n
        base_indices = []
        for j in range(n):
            for i in range(m):
                base_indices.append([i * int(r / m), j * int(t / n)])
        base_indices = tuple(reversed(base_indices))
        logging.debug("base_indices:\n" + str(base_indices))
        return base_indices

    def calculate_C(self, return_C, recv_index):
        """
        :param return_C: Ci calculated by workers
        :return: final C
        """
        r, s, t, m, n = self.r, self.s, self.t, self.m, self.n

        base_indices = self.calculate_base_indices()

        # list is 0 based but our workers, and Aenc, Benc matrices are 1 based
        # so we need to convert the list
        recv_var = tuple(map(lambda x: x + 1, recv_index))
        coeffs = np.zeros((r, t), dtype=NP_DATA_TYPE)
        logging.info("looping for %d" % (int(r / m) * int(t / n)))
        for i in range(int(r / m)):
            for j in range(int(t / n)):
                f_z = []
                for k in range(m * n):
                    f_z.append(return_C[recv_index[k]][i][j])
                lagrange_interpolate = interpolate.lagrange(recv_var, f_z)
                for index, lag_coef in enumerate(lagrange_interpolate):
                    coeffs[i + base_indices[index][0]][j + base_indices[index][1]] = lag_coef

        # logging.debug("coeffs: " + str(coeffs.shape))
        # logging.info("coeffs:\n" + str(coeffs))
        return coeffs

    @staticmethod
    def mapper(comm, barrier_enabled, straggling_enabled):
        spec_dict = comm.bcast(None)
        # spec_dict = [3073, 10, 500, 2125991977, 1, 7, 8]
        logging.debug("spec_dict " + str(spec_dict))
        r, t, s, F, n, m, N = spec_dict

        # Receive straggler information from the master
        straggler = comm.recv(source=0, tag=7)

        # Receive split input matrices from the master
        Ai = np.empty_like(np.matrix([[0] * int(r / m) for i in range(s)]), dtype=NP_DATA_TYPE)
        Bi = np.empty_like(np.matrix([[0] * int(t / n) for i in range(s)]), dtype=NP_DATA_TYPE)
        receive_A = comm.Irecv(Ai, source=0, tag=15)
        receive_B = comm.Irecv(Bi, source=0, tag=29)

        logging.debug("Ai[receiver] " + str(Ai.shape) + ", dtype=" + str(Ai.dtype))
        logging.debug("Bi[receiver] " + str(Bi.shape) + ", dtype=" + str(Bi.dtype))

        receive_A.wait()
        receive_B.wait()

        if barrier_enabled:
            comm.Barrier()

        wbp_received = time.time()

        # Start a separate thread to mimic background computation tasks if this is a straggler
        if straggling_enabled:
            if straggler == comm.rank:
                t = threading.Thread(target=loop)
                t.start()

        Ci = (Ai.getT() * Bi) % F
        logging.debug("r[" + str(comm.Get_rank()) + "] A:\n" + str(Ai)
                      + "\nB:\n" + str(Bi)
                      + "\nC:\n" + str(Ci))

        # print("Ci["+ str(comm.Get_rank()) +"]", Ci )
        wbp_done = time.time()
        logging.info("Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received))

        sC = comm.Isend([Ci, MPI_DATA_TYPE], dest=0, tag=42)
        sC.Wait()

    def polynomial_code(self):
        if self.comm.Get_rank() == 0:
            self.data_send()
            self.reducer()
