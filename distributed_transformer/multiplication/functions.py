# -*- coding: utf-8 -*-

"""Distributed multiplication autograd definitions."""

# Standard lib imports
from typing import Any, Callable, Union, Tuple, Sequence, Optional

# PyTorch imports
import torch
import torch.nn as nn
from torch import Tensor
import torch.autograd as autograd

# Local imports
from distributed_transformer.utils.comm import (
    get_rank, get_world_size, synchronize)

# Distributed imports
import horovod.torch as hvd


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
        partial_results = partial_results.split(1, dim=1)
        for rank, rank_col in enumerate(partial_results):
            result[rank * rows + row] = rank_col
    result = torch.cat(result, dim=-1)
    return result


def distributed_matmul_tn(left: Tensor, right: Tensor) -> Tensor:
    def get_splits(rank, mat, split_size):
        start = rank * split_size
        end = start + split_size
        return mat[:, :, start:end] if dims == 3 else mat[:, start:end]

    dims = left.dim()
    assert dims <= 3 and dims >= 2
    cols = left.size(dims - 1)
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
                rank_block[:, col] = all_cols
    return rank_block


class RightTransposeMultiplication(autograd.Function):
    @staticmethod
    def forward(ctx: Any, left: Tensor, right: Tensor) -> Tensor:
        ctx.save_for_backward(left, right)
        # self_mult = left.matmul(right.transpose(-2, -1))
        proj = distributed_matmul_nt(left, right)
        return proj

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> (Tensor, Tensor):
        # return super().backward(ctx, *grad_outputs)
        left, right = ctx.saved_tensors
        grad_left = grad_right = None
        return grad_left, grad_right
