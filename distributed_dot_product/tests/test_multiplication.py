
# -*- coding: utf-8 -*-

"""Tests for distributed multiplication operators."""

# Standard library imports
import functools

# PyTest imports
import pytest

# PyTorch imports
import torch

# Local imports
from distributed_dot_product.utils.comm import get_rank, get_world_size
from distributed_dot_product.multiplication.functions import (
    distributed_matmul_nt, distributed_matmul_tn, distributed_matmul_all)

# Distributed imports
import horovod.torch as hvd

LENGTH = 4
DIM = 6


def create_tensor(*sizes):
    product = functools.reduce(lambda x, y: x * y, sizes)
    seq_tensor = torch.arange(product)
    seq_tensor = seq_tensor.view(*sizes)
    return seq_tensor


MODES = {
    'NT': ((lambda world_size: create_tensor(1, LENGTH * world_size, DIM),
            lambda world_size: create_tensor(1, LENGTH * world_size, DIM),
            lambda world_size: create_tensor(world_size, LENGTH, DIM),
            lambda world_size: create_tensor(world_size, LENGTH, DIM)),
           (lambda x, y: torch.matmul(x, y.transpose(-1, -2)),
            functools.partial(distributed_matmul_nt, offset=2))),
    'TN': ((lambda world_size: create_tensor(1, LENGTH * world_size,
                                             LENGTH * world_size),
            lambda world_size: create_tensor(1, LENGTH * world_size, DIM),
            lambda world_size: create_tensor(world_size, LENGTH,
                                             LENGTH * world_size),
            lambda world_size: create_tensor(world_size, LENGTH, DIM)),
           (lambda x, y: torch.matmul(x.transpose(-1, -2), y),
            distributed_matmul_tn)),
    'FULL': ((lambda world_size: create_tensor(1, LENGTH * world_size,
                                               LENGTH * world_size),
              lambda world_size: create_tensor(1, LENGTH * world_size, DIM),
              lambda world_size: create_tensor(world_size, LENGTH,
                                               LENGTH * world_size),
              lambda world_size: create_tensor(world_size, LENGTH, DIM)),
             (lambda x, y: torch.matmul(x, y),
              functools.partial(distributed_matmul_all, offset=2)))
}


@pytest.fixture(scope='module')
def distributed_fixture():
    hvd.init()
    world_size = get_world_size()
    rank = get_rank()
    return world_size, rank


@pytest.fixture(scope='module')
def tensor_fixture(request, distributed_fixture):
    mode = request.param
    world_size, rank = distributed_fixture
    tensor_fn, funcs = MODES[mode]
    tensors = [f(world_size) for f in tensor_fn]
    gt_left, gt_right, test_left, test_right = tensors
    test_left = test_left[rank]
    test_right = test_right[rank]
    return (gt_left, gt_right), (test_left, test_right), funcs


@pytest.mark.parametrize('tensor_fixture', list(MODES.keys()), indirect=True)
def test_distributed_nt(tensor_fixture):
    gt_tensors, test_tensors, (gt_fn, test_fn) = tensor_fixture
    gt_result = gt_fn(*gt_tensors)
    test_result = test_fn(*test_tensors)
    assert gt_result == test_result
