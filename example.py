
# -*- coding: utf-8 -*-

"""Distributed multiplication usage example."""

import torch
import torch.nn as nn

from distributed_dot_product.utils.comm import get_rank, get_world_size
from distributed_dot_product.module import DistributedDotProductAttn

torch.manual_seed(111)

device = torch.device('cpu')

if torch.cuda.is_available():
    torch.cuda.set_device(get_rank())
    device = torch.device('cuda')

module = DistributedDotProductAttn(768, num_heads=2, offset=64)
module.to(device)

criterion = nn.MSELoss()

length = 4096
world_size = get_world_size()
x = torch.rand(1, length // world_size, 768, device=device)
y = torch.rand(1, length // world_size, 768, device=device)
mask = torch.zeros(1, length // world_size, length, device=device).bool()

out = module(x, x, x, mask)
loss = criterion(out, y)
loss.backward()
