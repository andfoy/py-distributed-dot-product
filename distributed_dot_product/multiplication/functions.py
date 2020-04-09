# -*- coding: utf-8 -*-

"""Distributed multiplication definitions."""

# Standard library imports
import os
import time
import functools

# PyTorch imports
import torch
from torch import Tensor

# Local imports
from distributed_dot_product.utils.comm import (
    get_rank, get_world_size, synchronize)

# Distributed imports
import horovod.torch as hvd

DEBUG = bool(os.environ.get('DISTRIBUTED_DOT_DEBUG', 0))


def measure(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        if DEBUG:
            print(f'{f.__name__} elapsed time: {time.time() - start_time}')
        return result
    return wrapper


@measure
def distributed_matmul_nt(left: Tensor, right: Tensor, offset=32) -> Tensor:
    """
    Multiply two sequence tensors to obtain the result of :math:`AB^T`.

    Left and right inputs must be 3-dimensional tensors of size
    :math:`* \times \frac{T}{N} \times D`, where :math:`T` is the total length,
    :math:`N` is the total number of processes available and :math:`D`, the
    dimension of the sequence. The result of this function is a tensor of size
    :math:`* \times \frac{T}{N} \times T`, that contain the result chunk for
    each process of the resulting operation.

    Inputs
    ------
    left: Tensor
        :math:`A` in :math:`AB^T`, must be of size
        :math:`* \times \frac{T}{N} \times D`.
    right: Tensor
        :math:`B` in :math:`AB^T`, must be of size
        :math:`* \times \frac{T}{N} \times D`.
    offset: int
        Number of chunks to communicate during each distributed step, it must
        be a factor of :math:`\frac{T}{N}`. This factor should be modified in
        order to reduce the total computing time at the expense of the memory
        used.

    Returns
    -------
    result: Tensor
        For each process, this function computes the corresponding segment
        of the operation :math:`A^T B`, of size
        :math:`* \times \frac{T}{N} \times T`.
    """
    synchronize()
    rows = left.size(-2)
    world_size = get_world_size()
    total_rows = rows * world_size

    prefix_size = tuple(left.size())[:-2]
    size = (world_size, left.size(-2), rows)
    size = prefix_size + size
    result = torch.empty(size, device=left.device)
    final_size = prefix_size + (left.size(-2), total_rows)

    for row in range(0, rows, offset):
        end_bound = row + offset
        current_row = right[..., row:end_bound, :]
        # [r0[row:end_bound], r1[row:end_bound], ..., rworld[row:end_bound]]
        # all_rows: world_size x offset x dim
        all_rows = hvd.allgather(current_row, name=f'scatter_rows_{row}')
        partial_results = left.matmul(all_rows.transpose(-1, -2))
        result[..., row:end_bound] = partial_results
    result = result.transpose(-3, -2).reshape(*final_size)
    return result


@measure
def distributed_matmul_tn(left: Tensor, right: Tensor) -> Tensor:
    """
    Multiply two sequence tensors to obtain the result of :math:`A^{T} B`.

    Left and right inputs must be 3-dimensional tensors, where the first one
    must be of size :math:`* \times \frac{T}{N} \times T` and the second one of
    size , where :math:`* \times \frac{T}{N} \times D`, where :math:`T` is the
    total length,  :math:`N` is the total number of processes available and
    :math:`D`, the dimension of the sequence. The result of this function is a
    tensor of size :math:`* \times \frac{T}{N} \times D`, that contain the
    result chunk for each process of the resulting operation.

    Inputs
    ------
    left: Tensor
        :math:`A` in :math:`A^T B`, must be of size
        :math:`* \times \frac{T}{N} \times T`
    right: Tensor
        :math:`B` in :math:`A^T B`, must be of size
        :math:`* \times \frac{T}{N} \times D`

    Returns
    -------
    result: Tensor
        For each process, this function computes the corresponding segment
        of the operation :math:`A^T B`, of size
        :math:`* \times \frac{T}{N} \times D`
    """
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    cols = left.size(-1)
    world_size = get_world_size()
    rank = get_rank()

    split_size = cols // world_size
    splits = left.split(split_size, -1)
    rank_block = None

    synchronize()
    for r in range(world_size):
        rank_split = splits[r]
        rank_multiplication = torch.matmul(rank_split.transpose(-1, -2), right)
        handle = hvd.allreduce_async(rank_multiplication,
                                     name=f'matmul_tn_{r}',
                                     op=hvd.Sum)
        if r == rank:
            rank_block = hvd.synchronize(handle)
    return rank_block.contiguous()


def distributed_matmul_block(left: Tensor, right: Tensor,
                             transpose: bool = False) -> Tensor:
    rank_block = torch.matmul(left, right)
    if transpose:
        rank_block = rank_block.tranpose(-1, -2)
    rank_block: Tensor = hvd.allreduce(rank_block, name='matmul_block')
    return rank_block


@measure
def distributed_matmul_all(left: Tensor, right: Tensor, offset=32) -> Tensor:
    """
    Multiply two sequence tensors to obtain the result of :math:`AB`.

    Left and right inputs must be 3-dimensional tensors, where the first one
    must be of size :math:`* \times \frac{T}{N} \times T` and the second one of
    size , where :math:`* \times \frac{T}{N} \times D`, where :math:`T` is the
    total length,  :math:`N` is the total number of processes available and
    :math:`D`, the dimension of the sequence. The result of this function is a
    tensor of size :math:`* \times \frac{T}{N} \times D`, that contain the
    result chunk for each process of the resulting operation.

    Inputs
    ------
    left: Tensor
        :math:`A` in :math:`AB`, must be of size
        :math:`* \times \frac{T}{N} \times T`
    right: Tensor
        :math:`B` in :math:`AB`, must be of size
        :math:`* \times \frac{T}{N} \times D`

    Returns
    -------
    result: Tensor
        For each process, this function computes the corresponding segment
        of the operation :math:`AB`, of size
        :math:`1 \times \frac{T}{N} \times D`
    """
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    cols = left.size(dims - 1)
    world_size = get_world_size()

    total_cols = right.size(-1)
    split_size = cols // world_size
    splits = torch.stack(left.split(split_size, -1), dim=0)
    left_sizes = tuple(left.size())
    size = (world_size,) + left_sizes[:-2] + (left.size(-2), total_cols)
    rank_block = torch.empty(*size, device=left.device)

    total_cols = right.size(-1)
    synchronize()
    for current_col in range(0, total_cols, offset):
        end_bound = current_col + offset
        col = right[..., current_col:end_bound]
        col = col.contiguous()
        all_cols = hvd.allgather(col.unsqueeze(0),
                                 name=f'matmul_all_{current_col}')
        # all_cols: torch.size([world_size, right.size(1), offset])
        block_result = torch.matmul(splits, all_cols)
        rank_block[..., current_col:end_bound] = block_result
    result = rank_block.sum(dim=0)
    return result
