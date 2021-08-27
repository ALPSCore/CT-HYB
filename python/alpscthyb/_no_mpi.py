import numpy as np

mpi4py = None
rank = 0
size = 1
MPI_TYPE_MAP = {}


def allreduce(data):
    """
    Return allreduced data
    """
    return data


def scatterv(send_buf, counts, displacements=None):
    return send_buf

def allgatherv(send_buf):
    return send_buf

def barrier(comm=None):
    pass

def split_idx(size, comm_size):
    if comm_size > 1:
        raise RuntimeError("comm_size must be for non_mpi mode!")
    return np.array([size]), np.array([0])


def get_slice(idx_size, comm=None):
    # get slice for 1D array of size idx_size
    return slice(0, idx_size)


def get_range(idx_size, comm=None):
    # get range for 1D array of size idx_size
    return range(idx_size)


def dist_w(w):
    """Distribute sampling frequencies/weights over MPI processes
    """
    return w