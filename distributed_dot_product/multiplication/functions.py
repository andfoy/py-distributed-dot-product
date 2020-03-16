# -*- coding: utf-8 -*-

"""Distributed multiplication definitions."""

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


def get_splits(rank, mat, split_size):
    start = rank * split_size
    end = start + split_size
    return mat[..., start:end]


def distributed_matmul_nt(left: Tensor, right: Tensor) -> Tensor:
    synchronize()
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    rows = left.size(dims - 2)
    world_size = get_world_size()
    # rank = get_rank()
    total_rows = rows * world_size
    result = [None for _ in range(0, total_rows)]
    for row in range(rows):
        current_row = right[:, row] if dims == 3 else right[row]
        # [r0[row], r1[row], ..., rworld[row]]
        # all_rows: Rxdim
        all_rows = hvd.allgather(current_row, name=f'scatter_rows_{row}')
        partial_results = left.matmul(all_rows.transpose(-1, -2))
        partial_results = partial_results.split(1, dim=-1)
        for rank, rank_col in enumerate(partial_results):
            result[rank * rows + row] = rank_col
    result = torch.cat(result, dim=-1)
    return result.contiguous()


def distributed_matmul_tn(left: Tensor, right: Tensor) -> Tensor:
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    cols = left.size(-1)
    world_size = get_world_size()
    rank = get_rank()

    total_cols = right.size(-1)
    split_size = cols // world_size

    # rank_result = left[:, :, start:end] if dims == 3 else left[:, start:end]
    splits = [get_splits(r, left, split_size) for r in range(world_size)]
    # rank_block = torch.matmul(left.transpose(-1, -2), splits[rank])
    size = ((left.size(0), left.size(1), right.size(-1))
            if dims == 3 else (left.size(0), right.size(-1)))
    rank_block = torch.zeros(*size, device=left.device)

    synchronize()
    for current_col in range(total_cols):
        col = right[:, :, current_col] if dims == 3 else right[:, current_col]
        for r in range(world_size):
            rank_split = splits[r]
            col_rank = torch.matmul(rank_split.transpose(-1, -2), col)
            all_cols = hvd.allreduce(col_rank, name=f'matmul_tn_{col}_{r}')
            if r == rank:
                rank_block[..., col] = all_cols
    return rank_block.contiguous()


def distributed_matmul_block(left: Tensor, right: Tensor,
                             transpose: bool = False) -> Tensor:
    rank_block = torch.matmul(left, right)
    if transpose:
        rank_block = rank_block.tranpose(-1, -2)
    rank_block: Tensor = hvd.allreduce(rank_block, name='matmul_block')
    return rank_block


def distributed_matmul_all(left: Tensor, right: Tensor) -> Tensor:
    dims = left.dim()
    assert dims <= 3 and dims >= 2
    cols = left.size(dims - 1)
    world_size = get_world_size()

    total_cols = right.size(-1)
    split_size = cols // world_size
    splits = [get_splits(r, left, split_size) for r in range(world_size)]
    size = ((left.size(0), left.size(1), right.size(-1))
            if dims == 3 else (left.size(0), right.size(-1)))
    rank_block = torch.zeros(*size, device=left.device)

    total_cols = right.size(-1)
    synchronize()
    for current_col in range(total_cols):
        col = right[..., current_col]
        col_result = rank_block[..., current_col]
        col = col.contiguous()
        all_cols = hvd.allgather(col, name=f'matmul_all_{current_col}')
        all_cols = all_cols.split(1, dim=0)
        for i, (split, rank_col) in enumerate(zip(splits, all_cols)):
            rank_result = torch.matmul(split, rank_col.unsqueeze(-1))
            rank_result = rank_result.squeeze(-1)
            col_result = col_result + rank_result
        rank_block[..., current_col] = col_result
    return rank_block
