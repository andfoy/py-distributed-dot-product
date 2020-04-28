# -*- coding: utf-8 -*-

"""
Distributed multiplication benchmark (time/memory).

The following benchmarks were run on a machine with 3 Quadro RTX 6000 GPUs,
each one with 24Gb of RAM.
"""

import os
import time
import json
import humanize
import argparse
import functools
import os.path as osp

import torch
from mpi4py import MPI

from distributed_dot_product.utils.comm import (
    synchronize, is_main_process, get_rank)
from distributed_dot_product.multiplication.functions import (
    distributed_matmul_all, distributed_matmul_nt, distributed_matmul_tn)

comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(
    description='Benchmark suit for distributed operations')
parser.add_argument('--mode', type=str, default='nt',
                    help='Benchmark mode')
parser.add_argument('--offset', type=int, default=1000,
                    help='Offset used to balance time/memory')
parser.add_argument('--scale', type=int, default=1,
                    help='Scale factor used for reduction')
parser.add_argument('--file', type=str, default="nt_benchmark.json",
                    help='JSON file used to append results')
args = parser.parse_args()


values = []
if osp.exists(args.file):
    values = json.load(open(args.file, 'r'))

torch.set_grad_enabled(False)
torch.manual_seed(111)

device = torch.device('cpu')

if torch.cuda.is_available():
    torch.cuda.set_device(get_rank())
    device = torch.device('cuda')


def measure(function, *args, **kwargs):
    torch.cuda.reset_max_memory_allocated()
    start_memory = torch.cuda.max_memory_allocated()
    start_time = time.time()
    y = function(*args, **kwargs)
    total_time = time.time() - start_time
    end_memory = torch.cuda.max_memory_allocated()
    print(f'{function.__name__} - Total time elapsed: {total_time}s')
    print(f'{function.__name__} - '
          f'Memory consumption: '
          f'{humanize.naturalsize(end_memory - start_memory)}')
    return y, total_time, end_memory - start_memory


def nt_benchmark():
    # Benchmark NT multiplication (local node)
    if is_main_process():
        xlarge = torch.rand(1, 75000 // args.scale, 768, device=device)
        y = xlarge.transpose(-1, -2)
        input_memory = torch.cuda.memory_allocated()
        print(f'Memory allocated by xlarge/y: '
            f'{humanize.naturalsize(input_memory)}')
        result, op_time, peak_memory = measure(torch.matmul, xlarge, y)
        del xlarge
        del y
        torch.cuda.empty_cache()
        output_memory = torch.cuda.memory_allocated()
        print(f'matmul_nt - Output memory consumption: '
            f'{humanize.naturalsize(output_memory)}')
        del result
        torch.cuda.empty_cache()

    # Benchmark TN multiplication (distributed)
    xsmall = torch.rand(1, 75000 // (3 * args.scale), 768, device=device)
    dist_input_size = torch.cuda.memory_allocated()
    print(f'Memory allocated by xsmall: '
        f'{humanize.naturalsize(dist_input_size)}')
    synchronize()
    result, dop_time, dpeak_memory = measure(
        distributed_matmul_nt, xsmall, xsmall, offset=1000)
    del xsmall
    torch.cuda.empty_cache()
    doutput_memory = torch.cuda.memory_allocated()
    print(f'distributed_matmul_nt - Output memory consumption: '
        f'{humanize.naturalsize(doutput_memory)}')
    del result
    torch.cuda.empty_cache()

    all_input_size = comm.gather(dist_input_size, root=0)
    all_op_time = comm.gather(dop_time, root=0)
    all_peak_memory = comm.gather(dpeak_memory, root=0)
    all_output_memory = comm.gather(doutput_memory, root=0)

    if is_main_process():
        avg_input_size = sum(all_input_size) / len(all_input_size)
        avg_op_time = sum(all_op_time) / len(all_op_time)
        avg_peak_memory = sum(all_peak_memory) / len(all_peak_memory)
        avg_output_memory = sum(all_output_memory) / len(all_output_memory)

        return (input_memory, output_memory, op_time, peak_memory,
                avg_input_size, avg_op_time, avg_peak_memory,
                avg_output_memory)


def all_benchmark():
    # Benchmark all multiplication (local node)
    if is_main_process():
        xlarge = torch.rand(1, 75000 // args.scale, 75000 // args.scale,
                            device=device)
        y = torch.rand(1, 75000 // args.scale, 768, device=device)
        input_memory = torch.cuda.memory_allocated()
        print(f'Memory allocated by xlarge/y: '
            f'{humanize.naturalsize(input_memory)}')
        result, op_time, peak_memory = measure(torch.matmul, xlarge, y)
        del xlarge
        del y
        torch.cuda.empty_cache()
        output_memory = torch.cuda.memory_allocated()
        print(f'matmul_nt - Output memory consumption: '
            f'{humanize.naturalsize(output_memory)}')
        del result
        torch.cuda.empty_cache()

    # Benchmark all multiplication (distributed)
    xsmall = torch.rand(1, 75000 // (3 * args.scale), 75000 // args.scale,
                        device=device)
    ysmall = torch.rand(1, 75000 // (3 * args.scale), 768, device=device)
    dist_input_size = torch.cuda.memory_allocated()
    print(f'Memory allocated by xsmall/ysmall: '
          f'{humanize.naturalsize(dist_input_size)}')
    synchronize()
    result, dop_time, dpeak_memory = measure(
        distributed_matmul_all, xsmall, ysmall, offset=args.offset)
    del xsmall
    del ysmall

    torch.cuda.empty_cache()
    doutput_memory = torch.cuda.memory_allocated()
    print(f'distributed_matmul_all - Output memory consumption: '
        f'{humanize.naturalsize(doutput_memory)}')
    del result
    torch.cuda.empty_cache()

    all_input_size = comm.gather(dist_input_size, root=0)
    all_op_time = comm.gather(dop_time, root=0)
    all_peak_memory = comm.gather(dpeak_memory, root=0)
    all_output_memory = comm.gather(doutput_memory, root=0)

    if is_main_process():
        avg_input_size = sum(all_input_size) / len(all_input_size)
        avg_op_time = sum(all_op_time) / len(all_op_time)
        avg_peak_memory = sum(all_peak_memory) / len(all_peak_memory)
        avg_output_memory = sum(all_output_memory) / len(all_output_memory)

        return (input_memory, output_memory, op_time, peak_memory,
                avg_input_size, avg_op_time, avg_peak_memory,
                avg_output_memory)


def tn_benchmark():
    # Benchmark tn multiplication (local node)
    if is_main_process():
        xlarge = torch.rand(1, 75000 // args.scale, 75000 // args.scale,
                            device=device)
        y = torch.rand(1, 75000 // args.scale, 768, device=device)
        input_memory = torch.cuda.memory_allocated()
        print(f'Memory allocated by xlarge/y: '
            f'{humanize.naturalsize(input_memory)}')
        result, op_time, peak_memory = measure(torch.matmul, xlarge, y)
        del xlarge
        del y
        torch.cuda.empty_cache()
        output_memory = torch.cuda.memory_allocated()
        print(f'matmul_nt - Output memory consumption: '
            f'{humanize.naturalsize(output_memory)}')
        del result
        torch.cuda.empty_cache()

    # Benchmark tn multiplication (distributed)
    xsmall = torch.rand(1, 75000 // (3 * args.scale), 75000 // args.scale,
                        device=device)
    ysmall = torch.rand(1, 75000 // (3 * args.scale), 768, device=device)
    dist_input_size = torch.cuda.memory_allocated()
    print(f'Memory allocated by xsmall/ysmall: '
          f'{humanize.naturalsize(dist_input_size)}')
    synchronize()
    result, dop_time, dpeak_memory = measure(
        distributed_matmul_tn, xsmall, ysmall)
    del xsmall
    del ysmall

    torch.cuda.empty_cache()
    doutput_memory = torch.cuda.memory_allocated()
    print(f'distributed_matmul_all - Output memory consumption: '
        f'{humanize.naturalsize(doutput_memory)}')
    del result
    torch.cuda.empty_cache()

    all_input_size = comm.gather(dist_input_size, root=0)
    all_op_time = comm.gather(dop_time, root=0)
    all_peak_memory = comm.gather(dpeak_memory, root=0)
    all_output_memory = comm.gather(doutput_memory, root=0)

    if is_main_process():
        avg_input_size = sum(all_input_size) / len(all_input_size)
        avg_op_time = sum(all_op_time) / len(all_op_time)
        avg_peak_memory = sum(all_peak_memory) / len(all_peak_memory)
        avg_output_memory = sum(all_output_memory) / len(all_output_memory)

        return (input_memory, output_memory, op_time, peak_memory,
                avg_input_size, avg_op_time, avg_peak_memory,
                avg_output_memory)


def main():
    test_funcs = {
        'nt': nt_benchmark,
        'all': all_benchmark,
        'tn': tn_benchmark
    }
    output = test_funcs[args.mode]()
    if is_main_process():
        (input_memory, output_memory, op_time, peak_memory, avg_input_size,
         avg_op_time, avg_peak_memory, avg_output_memory) = output

        output = {
            'input_memory': input_memory,
            'total_time': op_time,
            'peak_memory': peak_memory,
            'output_memory': output_memory,
            'distributed_input_memory': avg_input_size,
            'distributed_time': avg_op_time,
            'distributed_peak_memory': avg_peak_memory,
            'distributed_output_memory': avg_output_memory
        }

        values.append(output)
        json.dump(values, open(args.file, 'w'))
    synchronize()


if __name__ == '__main__':
    main()
