
# -*- coding: utf-8 -*-

"""Distributed multiplication autograd definitions."""

# Standard lib imports
from typing import Any, Callable, Union, Tuple, Sequence, Optional

# PyTorch imports
from torch import Tensor
import torch.autograd as autograd

# Local imports
from distributed_dot_product.multiplication.functions import (
    distributed_matmul_nt, distributed_matmul_tn, distributed_matmul_block,
    distributed_matmul_all
)


class RightTransposeMultiplication(autograd.Function):
    @staticmethod
    def forward(ctx: Any, left: Tensor, right: Tensor, offset: int) -> Tensor:
        ctx.save_for_backward(left, right)
        ctx.offset = offset
        # self_mult = left.matmul(right.transpose(-2, -1))
        proj = distributed_matmul_nt(left, right)
        return proj

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> (Tensor, Tensor, None):
        # return super().backward(ctx, *grad_outputs)
        left, right = ctx.saved_tensors
        offset = ctx.offset
        grad_left = grad_right = None
        grad_right = distributed_matmul_tn(output_grad, left)
        grad_left = distributed_matmul_all(output_grad, right, offset=offset)
        # grad_left = grad_left.transpose(-1, -2)
        return grad_left, grad_right, None


class FullMultiplication(autograd.Function):
    @staticmethod
    def forward(ctx: Any, left: Tensor, right: Tensor, offset: int) -> Tensor:
        ctx.save_for_backward(left, right)
        ctx.offset = offset
        result = distributed_matmul_all(left, right)
        return result

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> (Tensor, Tensor, None):
        left, right = ctx.saved_tensors
        offset = ctx.offset
        left_grad = distributed_matmul_nt(output_grad, right, offset=offset)
        right_grad = distributed_matmul_tn(left, output_grad)
        return left_grad, right_grad, None
