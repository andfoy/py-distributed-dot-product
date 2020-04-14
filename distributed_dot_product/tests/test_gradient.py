# -*- coding: utf-8 -*-

"""Tests for distributed multiplication operators (gradients)."""

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


@pytest.fixture(scope='function')
def tensor_fixture(request):
    x = torch.rand(1, LENGTH, DIM)
    y = torch.rand(1, LENGTH, DIM)
    mask = torch.zeros(1, LENGTH, LENGTH * get_world_size())

    def teardown():
        nonlocal x, y, mask
        del x
        del y
        del mask
        torch.cuda.empty_cache()

    request.addfinalizer(teardown)
    return x, y, mask.bool()


@pytest.fixture(scope='function', params=[1, 4])
def model_fixture(request):
    heads = request.param
    model = DistributedDotProductAttn(256, 256, 256, num_heads=heads)
    gt_model = DistributedDotProductAttn(256, 256, 256, num_heads=heads,
                                         distributed=False)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    gt_model.keys.weight.data = model.keys.weight.data.clone()
    gt_model.queries.weight.data = model.queries.weight.data.clone()
    gt_model.values.weight.data = model.values.weight.data.clone()
    gt_model.composition.weight.data = model.composition.weight.data.clone()

    def teardown():
        nonlocal model, gt_model
        del model
        del gt_model
        torch.cuda.empty_cache()

    request.addfinalizer(teardown)
    return model, gt_model


@pytest.fixture(scope='function', params=['cpu', 'cuda'])
def device_fixture(request):
    mode = request.param
    if mode == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available on this machine')
    device = torch.device(mode)
    return device


@pytest.mark.parametrize('tensors,models,device', [
    (pytest.lazy_fixture('tensor_fixture'),
     pytest.lazy_fixture('model_fixture'),
     pytest.lazy_fixture('device_fixture'))])
def test_gradient(tensors, models, device):
    k, q, mask = tensors
    k = k.to(device)
    q = q.to(device)
    mask = mask.to(device)

    k = k.requires_grad_(True)
    q = q.requires_grad_(True)

    model, gt_model = models
    model = model.to(device)
    gt_model = gt_model.to(device)

    distr_out = model(k, q, k, mask)
    reduction = distr_out.sum()
    reduction.backward()

    k_gt = hvd.allgather(k.detach())
    q_gt = hvd.allgather(q.detach())

    k_gt = k_gt.view(1, -1, k_gt.size(-1))
    q_gt = q_gt.view(1, -1, q_gt.size(-1))
    k_gt = k_gt.requires_grad_(True)
    q_gt = q_gt.requires_grad_(True)
    gt_mask = torch.zeros(1, k_gt.size(1), q_gt.size(1), device=k_gt.device)

    gt_out = gt_model(k_gt, q_gt, k_gt, gt_mask.bool())
    gt_reduction = gt_out.sum()
    gt_reduction.backward()

    k_grad = hvd.allgather(k.grad)
    k_grad = k_grad.view(1, -1, k_grad.size(-1))

    q_grad = hvd.allgather(q.grad)
    q_grad = q_grad.view(1, -1, q_grad.size(-1))

    assert torch.allclose(k_grad, k_gt.grad, atol=1e-5)
    assert torch.allclose(q_grad, q_gt.grad, atol=1e-5)

    for ((gt_name, gt_param), (name, param)) in zip(
            gt_model.named_parameters(), model.named_parameters()):
        assert gt_name == name
        gt_grad = gt_param.grad
        grad = hvd.allreduce(param.grad, op=hvd.Sum)
        assert torch.allclose(gt_grad, grad, atol=1e-5)
