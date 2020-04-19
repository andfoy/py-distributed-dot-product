# -*- coding: utf-8 -*-

"""Distributed multiplication benchmark (time/memory)."""

import time
import torch

from distributed_dot_product.utils.comm import synchronize, is_main_process
from distributed_dot_product.multiplication.functions import (
    distributed_matmul_all, distributed_matmul_nt, distributed_matmul_tn)


