import numpy as np

import mpi4py
from mpi4py import MPI

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

MPI_TYPE_MAP = {
    'int8': MPI.CHAR,
    'int16': MPI.SHORT,
    'int32': MPI.INT,
    'int64': MPI.LONG,
    'int128': MPI.LONG_LONG,
    'float32': MPI.FLOAT,
    'float64': MPI.DOUBLE,
    'bool': MPI.BOOL,
    'complex128': MPI.DOUBLE_COMPLEX
}


def allreduce(data):
    """
    Return allreduced data and a new token.
    See the document of allreduce for more details about tokens.
    """
    if np.isscalar(data):
        return _allreduce([data])[0]
    else:
        return _allreduce(data)

def _allreduce(data):
    MPI = mpi4py.MPI
    data = np.ascontiguousarray(data)
    recv_buf = np.empty_like(data)
    MPI.COMM_WORLD.Allreduce(data, recv_buf, MPI.SUM)
    return recv_buf

def scatterv(send_buf, counts, displacements=None):
    MPI = mpi4py.MPI
    comm = MPI.COMM_WORLD
    send_buf = np.ascontiguousarray(send_buf).ravel()
    rank = comm.Get_rank()
    recv_buf = np.empty(counts[rank], dtype=send_buf.dtype)
    MPI_TYPE = MPI_TYPE_MAP[str(send_buf.dtype)]
    if displacements is None:
        displacements = np.zeros(len(counts), dtype=int)
        displacements[1:] = np.cumsum(counts)[:-1]
    comm.Scatterv([send_buf, counts, displacements, MPI_TYPE], recv_buf, root=0)
    return recv_buf


def allgatherv(send_buf):
    MPI = mpi4py.MPI
    comm = MPI.COMM_WORLD
    send_buf = np.ascontiguousarray(send_buf).ravel()
    sizes = comm.allgather(send_buf.size)
    offsets = np.hstack([0, np.cumsum(sizes)])[0:comm.Get_size()]
    recv_buf = np.empty(np.sum(sizes), dtype=send_buf.dtype)
    MPI_TYPE = MPI_TYPE_MAP[str(send_buf.dtype)]
    comm.Allgatherv([send_buf, MPI_TYPE], [recv_buf, sizes, offsets, MPI_TYPE])
    return recv_buf

def barrier(comm=None):
    MPI = mpi4py.MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm.barrier()


def split_idx(size, comm_size):
    """Compute sizes and offsets for splitting a 1D array

    Parameters
    ----------
    size : Int
        Length of array
    comm_size : Int
        Number of MPI  processes

    Returns
    -------
    sizes : Int[:]
        Sizes for each MPI processes
    offsets : Int[:]
        Offsets for each MPI processes
    """
    base = size // comm_size
    leftover = int(size % comm_size)

    sizes = np.ones(comm_size, dtype=int) * base
    sizes[:leftover] += 1

    offsets = np.zeros(comm_size, dtype=int)
    offsets[1:] = np.cumsum(sizes)[:-1]

    return sizes, offsets

def get_slice(idx_size, comm=None):
    # get slice for 1D array of size idx_size
    if comm is None:
        comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    sizes, offsets = split_idx(idx_size, size)
    return slice(offsets[rank], offsets[rank]+sizes[rank])


def get_range(idx_size, comm=None):
    # get range for 1D array of size idx_size
    if comm is None:
        comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    sizes, offsets = split_idx(idx_size, size)
    return range(offsets[rank], offsets[rank]+sizes[rank])

def dist_w(w):
    """Distribute sampling frequencies/weights over MPI processes
    """
    if isinstance(w, tuple):
        nw = len(w[0])
        return tuple((x[get_range(nw)] for x in w))
    elif isinstance(w, np.ndarray):
        if w.ndim > 1:
            raise ValueError("ndim > 1 is not supported!")
        return w[get_range(len(w))]
    else:
        raise ValueError("Invalid w!")

