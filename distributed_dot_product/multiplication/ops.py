
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
        grad_right = distributed_matmul_tn(output_grad, left)
        grad_left = distributed_matmul_all(output_grad, right)
        grad_left = grad_left.transpose(-1, -2)
        return grad_left, grad_right


class FullMultiplication(autograd.Function):
    @staticmethod
    def forward(ctx: Any, left: Tensor, right: Tensor) -> Tensor:
        ctx.save_for_backward(left, right)
        result = distributed_matmul_all(left, right)
        return result

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> (Tensor, Tensor):
        left, right = ctx.saved_tensors
        left_grad = distributed_matmul_nt(output_grad, right)
        right_grad = distributed_matmul_tn(left, output_grad)
        return left_grad, right_grad
