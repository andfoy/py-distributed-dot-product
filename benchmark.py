# -*- coding: utf-8 -*-

"""
Distributed multiplication benchmark (time/memory).

The following benchmarks were run on a machine with 3 Quadro RTX 6000 GPUs,
each one with 24Gb of RAM.
"""

import time
import humanize
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


def measure(function, x, y, **kwargs):
    torch.cuda.reset_max_memory_allocated()
    start_memory = torch.cuda.max_memory_allocated()
    x = x()
    y = y()
    initial_size = torch.cuda.memory_allocated()
    print(f'Memory allocated by inputs: {humanize.naturalsize(initial_size)}')

    start_time = time.time()
    z = function(x, y)
    total_time = time.time() - start_time

    end_memory = torch.cuda.max_memory_allocated()

    del x
    del y
    torch.cuda.empty_cache()
    result_memory = torch.cuda.memory_allocated()

    print(f'{function.__name__} - Total time elapsed: {total_time}s')
    print(f'{function.__name__} - '
          f'Memory consumption (Peak): {humanize.naturalsize(end_memory)}')
    print(f'{function.__name__} - '
          f'Memory consumption (Result): {humanize.naturalsize(result_memory)}')
    return z


# Benchmark NT multiplication (local node)
if is_main_process():
    xlarge = lambda: torch.rand(1, 75000, 768, device=device)
    y = lambda: torch.rand(1, 768, 75000, device=device)
    # initial_size = torch.cuda.memory_allocated()
    # print(f'Memory allocated by xlarge/y: {humanize.naturalsize(initial_size)}')
    result = measure(torch.matmul, xlarge, y)
    del xlarge
    del y
    del result
    torch.cuda.empty_cache()

# Benchmark TN multiplication (distributed)
xsmall = lambda: torch.rand(1, 75000 // 3, 768, device=device)
synchronize()
result = measure(distributed_matmul_nt,
                 xsmall, xsmall, offset=1000)
del xsmall
del result
torch.cuda.empty_cache()
