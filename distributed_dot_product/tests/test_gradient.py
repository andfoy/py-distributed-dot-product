# -*- coding: utf-8 -*-

"""Tests for distributed multiplication operators (gradients)."""

# Standard library imports
import functools

# PyTest imports
import pytest

# PyTorch imports
import torch

# Local imports
from distributed_dot_product.utils.comm import get_rank, get_world_size
from distributed_dot_product.module import DistributedDotProductAttn

# Distributed imports
import horovod.torch as hvd

LENGTH = 18
DIM = 256

if torch.cuda.is_available():
    torch.cuda.set_device(get_rank())


@pytest.fixture
def tensor_fixture():
    x = torch.rand(1, LENGTH, DIM)
    x = x.requires_grad_(True)
    y = torch.rand(1, LENGTH, DIM)
    y = y.requires_grad_(True)
    mask = torch.zeros(1, LENGTH, LENGTH)
    return x, y, mask


@pytest.fixture(params=[1, 4])
def model_fixture(request):
    heads = request.param
    model = DistributedDotProductAttn(256, 256, 256, num_heads=heads)
    gt_model = DistributedDotProductAttn(256, 256, 256, num_heads=heads,
                                         distributed=False)
    return model, gt_model


@pytest.fixture(params=['cpu', 'cuda'])
def device_fixture(request):
    mode = request.param
    if mode == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available on this machine')
    device = torch.device(mode)
    return device


@pytest.mark.parametrize('tensors,models,device', [
    pytest.lazy_fixture('tensor_fixture'),
    pytest.lazy_fixture('model_fixture'),
    pytest.lazy_fixture('device_fixture')])
def test_gradient(tensors, models, device):
    k, q, mask = tensors
    k = k.to(device)
    q = q.to(device)
    mask = mask.to(device)

    model, gt_model = models
    model = model.to(device)
    gt_model = gt_model.to(device)

    distr_out = model(k, q, k, mask)
    distr_out.backward()

    k_gt = hvd.allgather(k.detach())
    q_gt = hvd.allgather(q.detach())

    k_gt = k_gt.view(1, -1, k_gt.size(-1))
    q_gt = q_gt.view(1, -1, q_gt.size(-1))
    k_gt = k_gt.requires_grad_(True)
    q_gt = q_gt.requires_grad_(True)
    gt_mask = torch.zeros(1, k_gt.size(1), q_gt.size(1), device=k_gt.device)
    gt_out = gt_model(k_gt, q_gt, k_gt, gt_mask)
    gt_out.backward()
    assert False
