# -*- coding: utf-8 -*-

"""
Language modelling example.

Sample training and inference script for training a Transformer in the Enwik8
dataset.

Taken from https://github.com/asappresearch/sru/tree/master/language_model
"""

# Standard library imports
import os
import sys
import time
import math
import copy
import random
import argparse
from typing import Optional, List

# Third party imports
import numpy as np
from tensorboardX import SummaryWriter

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Distributed imports
import horovod.torch as hvd

# Local imports
from distributed_dot_product.module import DistributedDotProductAttn
from distributed_dot_product.utils.comm import (
    get_rank, get_world_size, comm, MPI)

Tensor = torch.Tensor
device = torch.device('cpu')

# Reset random seed
torch.manual_seed(111)
np.random.seed(111)

if torch.cuda.is_available():
    torch.cuda.set_device(get_rank())
    device = torch.device('cuda')
    torch.cuda.manual_seed(111)


# ---------------------- Data loading functions -------------------------------

def read_corpus(path, num_test_symbols=5000000):
    raw_data = open(path).read()
    raw_data = np.fromstring(raw_data, dtype=np.uint8)
    unique, data = np.unique(raw_data, return_inverse=True)
    train_data = data[: -2 * num_test_symbols]
    valid_data = data[-2 * num_test_symbols: -num_test_symbols]
    test_data = data[-num_test_symbols:]
    return train_data, valid_data, test_data, unique


def create_batches(data_ids, batch_size):
    N = len(data_ids)
    L = ((N - 1) // batch_size) * batch_size
    x = np.copy(data_ids[:L].reshape(batch_size, -1).T)
    y = np.copy(data_ids[1:L + 1].reshape(batch_size, -1).T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.contiguous(), y.contiguous()
    x, y = x.to(device), y.to(device)
    return x, y

# ------------------------------- Models --------------------------------------

ACTIVATIONS = {
    'relu': F.relu,
    'gelu': F.gelu,
    'glu': F.glu
}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TimeEmbeddings(nn.Module):
    def __init__(self, dim, max_steps=10000, dropout=0.1):
        super(TimeEmbeddings, self).__init__()
        pe = torch.zeros(max_steps, dim)
        time = torch.arange(0, max_steps).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(time.float() * div_term)
        pe[:, 1::2] = torch.cos(time.float() * div_term)
        # self.pe = nn.Parameter(pe)
        self.pe = pe
        # self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        if self.pe.device != emb.device:
            self.pe = self.pe.to(emb.device)
        emb = emb + self.pe[:emb.size(1)]
        emb = self.dropout(emb)
        return emb


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model : int, nhead: int, ff_size: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu', normalize_before: bool = False,
                 distributed: bool = False):
        super().__init__()
        self.self_attn = DistributedDotProductAttn(
            d_model, num_heads=nhead, distributed=distributed)
        self.positional_embeddings = TimeEmbeddings(d_model, dropout=dropout)
        self.linear1 = nn.Linear(d_model, ff_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_size, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = ACTIVATIONS[activation]

        self.normalize_before = normalize_before
        self.distributed = distributed

    def forward_post(self,
                     src: Tensor,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.positional_embeddings(src)
        if src_mask is None:
            B, T, _ = src.size()
            src_mask = torch.zeros(B, T, T * get_world_size(), device=device)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src: Tensor, src_mask: Optional[Tensor] = None):
        src2 = self.norm1(src)
        # q = k = self.with_pos_embed(src2, pos)
        q = k = self.positional_embeddings(src2)
        if src_mask is None:
            B, T, _ = src.size()
            src_mask = torch.zeros(B, T, T * get_world_size(), device=device)
        src2 = self.self_attn(q, k, src2, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Model(nn.Module):
    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args
        # self.emb_size = (
        #     len(words) if len(words) < args.emb_size else args.emb_size)
        self.emb_size = args.emb_size
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = nn.Embedding(len(words), self.emb_size)
        self.output_size = len(words)
        self.distributed = args.distributed
        layer = TransformerEncoderLayer(self.emb_size, args.num_heads,
                                        args.ff_size, dropout=args.dropout,
                                        distributed=self.distributed)
        self.transformer = TransformerEncoder(layer, self.depth)
        self.output_layer = nn.Linear(self.emb_size, self.output_size)
        self.init_weights()

    def init_weights(self, val_range=None):
        #val_range = val_range or (3.0/self.n_d)**0.5
        params = (list(self.embedding_layer.parameters()) +
                  list(self.output_layer.parameters()))
        for p in params:
            if p.dim() > 1:  # matrix
                val = val_range or (3.0 / p.size(0)) ** 0.5
                p.data.uniform_(-val, val)
            else:
                p.data.zero_()

    def forward(self, x):
        emb = self.drop(self.embedding_layer(x))
        output = self.transformer(emb)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output


def calc_norm(lis):
    l2_sum = sum(x.norm() ** 2 for x in lis)
    return l2_sum ** 0.5


def eval_model(model, valid):
    with torch.no_grad():
        model.eval()
        args = model.args
        batch_size = valid[0].size(1)
        total_loss = 0.0
        unroll_size = args.unroll_size
        criterion = nn.CrossEntropyLoss(size_average=False)
        # hidden = model.init_hidden(batch_size)
        N = (len(valid[0]) - 1) // unroll_size + 1
        for i in range(N):
            x = valid[0][i * unroll_size:(i + 1) * unroll_size]
            y = valid[1][i * unroll_size:(i + 1) * unroll_size].view(-1)
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss = hvd.allreduce(loss, op=hvd.Sum)
            total_loss += loss.item()
        avg_loss = total_loss / comm.allreduce(valid[1].numel(), op=MPI.SUM)
        ppl = np.exp(avg_loss)
        model.train()
    return ppl, avg_loss

def copy_model(model):
    states = model.state_dict()
    for k in states:
        v = states[k]
        states[k] = v.clone()
    return states


def main(args):
    train, dev, test, words  = read_corpus(args.data)
    log_path = "{}_{}".format(args.log, random.randint(1,100))
    train_writer = SummaryWriter(log_dir=log_path + "/train")
    dev_writer = SummaryWriter(log_dir=log_path + "/dev")

    model = Model(words, args)
    model.to(device)

    compression = (hvd.Compression.fp16
                   if args.fp16_allreduce else hvd.Compression.none)

    print(model)
    sys.stdout.write("vocab size: {}\n".format(
        model.emb_size
    ))
    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))
    sys.stdout.write("\n")

    dev_, test_ = dev, test
    train = create_batches(train, args.batch_size)
    dev = create_batches(dev, args.batch_size)
    test = create_batches(test, args.batch_size)
    lr = (args.lr if not args.noam else
          args.lr / ( args.emb_size ** 0.5 ) / ( args.warmup_steps ** 1.5 ))

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = lr,
        weight_decay = args.weight_decay
    )

    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    plis = [p for p in model.parameters() if p.requires_grad]
    niter = 1
    unchanged = 0
    best_dev = 1e+8
    unroll_size = args.unroll_size
    batch_size = args.batch_size
    N = (len(train[0]) - 1) // unroll_size + 1
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.max_epoch):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        for i in range(N):
            x = train[0][i * unroll_size:(i + 1) * unroll_size]
            y = train[1][i * unroll_size:(i + 1) * unroll_size].view(-1)

            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm(
                    model.parameters(), args.clip_grad)
            optimizer.step()

            if (niter - 1) % 100 == 0:
                #sys.stdout.write("\r{}".format(niter))
                #sys.stdout.flush()
                #train_writer.add_scalar('loss', loss.data[0], niter)
                train_writer.add_scalar('loss', loss.item(), niter)
                train_writer.add_scalar('pnorm',
                    calc_norm([x.data for x in plis]),
                    niter
                )
                train_writer.add_scalar('gnorm',
                    calc_norm([x.grad for x in plis]),
                    niter
                )

            if niter % args.log_period == 0 or i == N - 1 and get_rank() == 0:
                elapsed_time = (time.time() - start_time) / 60.0
                dev_ppl, dev_loss = eval_model(model, dev)
                sys.stdout.write("\rIter={}  lr={:.5f}  train_loss={:.4f} "
                                 " dev_loss={:.4f} dev_bpc={:.2f}\teta={:.1f}m"
                                 "\t[{:.1f}m]\n".format(
                                    niter,
                                    optimizer.param_groups[0]['lr'],
                                    loss.item(),  # loss.data[0],
                                    dev_loss,
                                    np.log2(dev_ppl),
                                    elapsed_time*N/(i+1),
                                    elapsed_time
                                ))
                if dev_ppl < best_dev:
                    best_dev = dev_ppl
                    checkpoint = copy_model(model)
                sys.stdout.write("\n")
                sys.stdout.flush()
                dev_writer.add_scalar('loss', dev_loss, niter)
                dev_writer.add_scalar('bpc', np.log2(dev_ppl), niter)

            niter += 1
            if args.noam:
                if niter >= args.warmup_steps:
                    lr = args.lr / (args.emb_size ** 0.5) / (niter ** 0.5)
                else:
                    lr = (args.lr / (args.emb_size ** 0.5) /
                          (args.warmup_steps ** 1.5)) * niter
                optimizer.param_groups[0]['lr'] = lr

    train_writer.close()
    dev_writer.close()

    if get_rank() == 0:
        model.load_state_dict(checkpoint)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    dev = create_batches(dev_, 1)
    test = create_batches(test_, 1)
    dev_ppl, dev_loss = eval_model(model, dev)
    test_ppl, test_loss = eval_model(model, test)
    sys.stdout.write("dev_bpc={:.3f}  test_bpc={:.3f}\n".format(
        np.log2(dev_ppl), np.log2(test_ppl)
    ))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--log", type=str, required=True)
    argparser.add_argument("--noam", action="store_true")
    argparser.add_argument("--warmup_steps", type=int, default=32000)
    argparser.add_argument("--layer_norm", action="store_true")
    argparser.add_argument("--lstm", action="store_true")
    argparser.add_argument("--data", type=str, required=True, help="training file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=128)
    # argparser.add_argument("--unroll_size", type=int, default=100)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--emb_size", type=int, default=1024)
    argparser.add_argument("--ff_size", type=int, default=2048)
    argparser.add_argument('--num_heads', type=int, default=8)
    # argparser.add_argument("--n_d", "--d", type=int, default=1024)
    argparser.add_argument("--n_proj", type=int, default=0)
    argparser.add_argument("--dropout", type=float, default=0.2,
        help="dropout probability"
    )
    argparser.add_argument("--bias", type=float, default=-3,
        help="intial bias of highway gates",
    )
    argparser.add_argument("--depth", type=int, default=6)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--weight_decay", type=float, default=1e-7)
    argparser.add_argument("--clip_grad", type=float, default=0.3)
    argparser.add_argument("--log_period", type=int, default=100000)

    args = argparser.parse_args()
    args.distributed = get_world_size() > 1
    print(args)
    main(args)
