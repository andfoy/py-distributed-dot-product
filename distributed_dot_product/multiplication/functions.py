# -*- coding: utf-8 -*-

"""Distributed multiplication definitions."""

# Standard library imports
import time
import functools

# PyTorch imports
import torch
# import torch.nn as nn
from torch import Tensor
# import torch.autograd as autograd

# Local imports
from distributed_dot_product.utils.comm import (
    get_rank, get_world_size, synchronize)

# Distributed imports
import horovod.torch as hvd


def measure(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        print(f'{f.__name__} elapsed time: {time.time() - start_time}')
        return result
    return wrapper


def get_splits(rank, mat, split_size):
    start = rank * split_size
    end = start + split_size
    return mat[..., start:end]


@measure
def distributed_matmul_nt(left: Tensor, right: Tensor) -> Tensor:
    synchronize()
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    rows = left.size(dims - 2)
    world_size = get_world_size()
    # rank = get_rank()
    total_rows = rows * world_size
    # result = [None for _ in range(0, total_rows)]
    offset = 32
    result = torch.empty((world_size, left.size(1), rows),
                         device=left.device)
    for row in range(0, rows, offset):
        end_bound = row + offset
        current_row = (right[:, row:end_bound]
                       if dims == 3 else right[row:end_bound])
        # [r0[row:end_bound], r1[row:end_bound], ..., rworld[row:end_bound]]
        # all_rows: world_size x offset x dim
        all_rows = hvd.allgather(current_row, name=f'scatter_rows_{row}')
        partial_results = left.matmul(all_rows.transpose(-1, -2))
        result[:, :, row:end_bound] = partial_results
        # partial_results = partial_results.split(1, dim=-1)
        # for rank, rank_col in enumerate(partial_results):
        #     result[rank * rows + row] = rank_col
    result = result.transpose(0, 1).reshape(left.size(0),
                                            left.size(1), total_rows)
    # result = torch.cat(result, dim=-1)
    return result


@measure
def distributed_matmul_tn(left: Tensor, right: Tensor) -> Tensor:
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    cols = left.size(-1)
    world_size = get_world_size()
    rank = get_rank()

    total_cols = right.size(-1)
    split_size = cols // world_size

    # rank_result = left[:, :, start:end] if dims == 3 else left[:, start:end]
    # splits = [get_splits(r, left, split_size) for r in range(world_size)]
    splits = left.split(split_size, -1)
    # rank_block = torch.matmul(left.transpose(-1, -2), splits[rank])
    size = ((left.size(0), left.size(1), right.size(-1))
            if dims == 3 else (left.size(0), right.size(-1)))
    rank_block = torch.empty(*size, device=left.device)

    synchronize()
    # for current_col in range(total_cols):
    #     col = right[:, :, current_col] if dims == 3 else right[:, current_col]
    #     for r in range(world_size):
    #         rank_split = splits[r]
    #         col_rank = torch.matmul(rank_split.transpose(-1, -2),
    #                                 col.unsqueeze(-1))
    #         col_rank = col_rank.squeeze(-1)
    #         all_cols = hvd.allreduce(col_rank, name=f'matmul_tn_{col}_{r}')
    #         if r == rank:
    #             rank_block[..., current_col] = all_cols
    #         synchronize()
    for r in range(world_size):
        rank_split = splits[r]
        rank_multiplication = torch.matmul(rank_split.transpose(-1, -2), right)
        handle = hvd.allreduce_async(rank_multiplication,
                                     name=f'matmul_tn_{r}')
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
def distributed_matmul_all(left: Tensor, right: Tensor) -> Tensor:
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    cols = left.size(dims - 1)
    world_size = get_world_size()

    total_cols = right.size(-1)
    split_size = cols // world_size
    # splits = [get_splits(r, left, split_size) for r in range(world_size)]
    splits = torch.cat(left.split(split_size, -1), dim=0)

    # size = ((left.size(0), left.size(1), right.size(-1))
    #         if dims == 3 else (left.size(0), right.size(-1)))
    size = (world_size, left.size(1), total_cols)
    rank_block = torch.empty(*size, device=left.device)

    total_cols = right.size(-1)
    synchronize()
    offset = 32
    for current_col in range(0, total_cols, offset):
        end_bound = current_col + offset
        col = right[..., current_col:end_bound]
        col = col.contiguous()
        # col_result = rank_block[..., current_col:end_bound]
        all_cols = hvd.allgather(col, name=f'matmul_all_{current_col}')
        # all_cols: torch.size([world_size, right.size(1), offset])
        block_result = torch.matmul(splits, all_cols)
        rank_block[..., current_col:end_bound] = block_result
        # all_cols = all_cols.split(1, dim=0)
        # for i, (split, rank_col) in enumerate(zip(splits, all_cols)):
        #     rank_result = torch.matmul(split, rank_col.unsqueeze(-1))
        #     rank_result = rank_result.squeeze(-1)
        #     col_result = col_result + rank_result
        # rank_block[..., current_col] = col_result
    result = rank_block.sum(dim=0).unsqueeze(0)
    # result = rank_block.transpose(0, 1).reshape(left.size(0),
    #                                             left.size(1),
    #                                             right.size(-1))
    return result
