# -*- coding: utf-8 -*-

# Standard lib imports
import math

# PyTorch imports
import torch
import torch.nn as nn
from torch import Tensor

# Horovod imports
import horovod.torch as hvd

# Local imports
from distributed_dot_product.multiplication.ops import (
    RightTransposeMultiplication, FullMultiplication)

hvd.init()


class DistributedDotProductAttn(nn.Module):
    def __init__(self, key_dim: int, value_dim: int = None,
                 query_dim: int = None, num_heads: int = 1,
                 add_bias: bool = False, offset: int = 32):
        super(DistributedDotProductAttn, self).__init__()
        assert key_dim % num_heads == 0
        value_dim = value_dim if value_dim is not None else key_dim
        query_dim = query_dim if query_dim is not None else key_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.offset = offset
        self.dim = key_dim // self.num_heads
        self.keys = nn.Linear(key_dim, key_dim, bias=add_bias)
        self.queries = nn.Linear(query_dim, key_dim, bias=add_bias)
        self.values = nn.Linear(value_dim, value_dim, bias=add_bias)
        self.composition = nn.Linear(value_dim, value_dim)

    def forward(self, keys: Tensor, queries: Tensor, values: Tensor,
                attn_mask: Tensor) -> Tensor:
        keys = self.keys(keys)
        queries = self.queries(queries)
        values = self.values(values)

        if self.num_heads > 1:
            keys = keys.view(*keys.size()[:-1], self.num_heads, self.dim)
            queries = queries.view(*queries.size()[:-1], self.num_heads,
                                   self.dim)
            values = values.view(*values.size()[:-1], self.num_heads, self.dim)

            keys = keys.transpose(-2, -3)
            values = values.transpose(-2, -3)
            queries = queries.transpose(-2, -3)

        projection = RightTransposeMultiplication.apply(keys, queries,
                                                        self.offset)
        projection = projection / math.sqrt(self.dim)
        projection = projection.masked_fill(attn_mask, -float('inf'))
        attn = torch.softmax(projection, dim=-1)
        outputs = FullMultiplication.apply(attn, values, self.offset)
        if self.num_heads > 1:
            outputs = outputs.transpose(-3, -2)
            outputs = outputs.reshape(*outputs.size()[:-2], self.value_dim)
        outputs = self.composition(outputs)
        return outputs
