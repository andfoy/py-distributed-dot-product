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


class RightTransposeMultiplication(autograd.Function):
    @staticmethod
    def forward(ctx: Any, left: Tensor, right: Tensor):
        ctx.save_for_backward(left, right)
        # self_mult = left.matmul(right.transpose(-2, -1))
        synchronize()
        proj = distributed_matmul_nt(left, right)
        return proj

    @staticmethod
    def backward(ctx, output_grad):
        # return super().backward(ctx, *grad_outputs)
        left, right = ctx.saved_tensors
        grad_left = grad_right = None
        return grad_left, grad_right
