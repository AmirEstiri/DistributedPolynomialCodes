from mpi4py import MPI

comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.Get_rank()
size = comm.Get_size()
print("Hello world from processor %s, rank %d out of %d processors\n",
       name, rank, size)
