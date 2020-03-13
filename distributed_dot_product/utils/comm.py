# -*- coding: utf-8 -*-

from mpi4py import MPI
import horovod.torch as hvd

hvd.init()

assert hvd.mpi_threads_supported()
assert hvd.size() == MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD


def get_world_size():
    return hvd.size()


def get_rank():
    return hvd.rank()


def is_main_process():
    return hvd.rank() == 0


def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training
    """
    comm.Barrier()
