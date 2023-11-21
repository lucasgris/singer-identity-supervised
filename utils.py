#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
    def __init__(self, log_dir):
        self.train_writer = SummaryWriter(os.path.join(log_dir, "train"))
        self.valid_writer = SummaryWriter(os.path.join(log_dir, "valid"))

    def train_log_step(self, name, value, step):
        self.train_writer.add_scalar(f"{name}/step", value, step)

    def train_log_epoch(self, name, value, epoch):
        self.train_writer.add_scalar(f"{name}/epoch", value, epoch)

    def valid_log_step(self, name, value, step):
        self.valid_writer.add_scalar(f"{name}/step", value, step)

    def valid_log_epoch(self, name, value, epoch):
        self.valid_writer.add_scalar(f"{name}/epoch", value, epoch)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)