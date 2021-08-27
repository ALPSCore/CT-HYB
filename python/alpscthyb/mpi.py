import os

_mpi_imported = "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ

def is_on():
    return _mpi_imported

if _mpi_imported:
    from ._mpi import *
else:
    from ._no_mpi import *