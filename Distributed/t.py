from mpi4py import MPI

comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.Get_rank()
size = comm.Get_size()
print("name, rank, size:",
       name, rank, size)
