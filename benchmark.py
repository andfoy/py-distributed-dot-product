# -*- coding: utf-8 -*-

"""
Distributed multiplication benchmark (time/memory).

The following benchmarks were run on a machine with 3 Quadro RTX 6000 GPUs,
each one with 24Gb of RAM.
"""

import time
import functools

import torch

from distributed_dot_product.utils.comm import (
    synchronize, is_main_process, get_rank)
from distributed_dot_product.multiplication.functions import (
    distributed_matmul_all, distributed_matmul_nt, distributed_matmul_tn)


torch.set_grad_enabled(False)
torch.manual_seed(111)

device = torch.device('cpu')

if torch.cuda.is_available():
    torch.cuda.set_device(get_rank())
    device = torch.device('cuda')


def measure(function, *args, **kwargs):
    start_time = time.time()
    start_memory = torch.cuda.max_memory_allocated()
    y = function(*args, **kwargs)
    total_time = time.time() - start_time
    end_memory = torch.cuda.max_memory_allocated()
    print(f'{function.__name__} - Total time elapsed: {total_time}')
    print(f'{function.__name__} - '
          f'Memory consumption: {(end_memory - start_memory) / (1024.0 ** 2)}')
    return y


# Benchmark NT multiplication (local node)
if is_main_process():
    xlarge = torch.rand(1, 75000, 768, device=device)
    y = xlarge.transpose(-1, -2)
    result = measure(torch.matmul, xlarge, y)
    del xlarge
    del y
    del result
    torch.cuda.empty_cache()

synchronize()
# Benchmark TN multiplication (distributed)
xsmall = torch.rand(1, 75000 // 3, 768, device=device)
result = measure(distributed_matmul_nt, xsmall, xsmall, offset=500)
del xsmall
del result
torch.cuda.empty_cache()
