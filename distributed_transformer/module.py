# -*- coding: utf-8 -*-

# Standard lib imports
from typing import Any, Callable, Union, Tuple, Sequence, Optional

# PyTorch imports
import torch
import torch.nn as nn
from torch import Tensor
import torch.autograd as autograd

# Horovod imports
import horovod.torch as hvd

hvd.init()


class DistributedDotProductAttn(autograd.Function):
    @staticmethod
    def forward(ctx: Any, keys: Tensor, queries: Tensor, values: Tensor,
                k_mat: Tensor, v_mat: Tensor, q_mat: Tensor):
        ctx.save_for_backward(keys, queries, values, k_mat, v_mat, q_mat)
        k = keys.matmul(k_mat.t())
        q = queries.matmul(q_mat.t())
        # v = values.matmul(v_mat.t())

