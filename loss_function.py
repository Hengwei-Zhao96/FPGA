#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/28 16:09
# @Author : Hw-Zhao
# @Site : 
# @File : loss_function.py
# @Software: PyCharm
import torch.nn.functional as F


def loss(x, y, weight):
    losses = F.cross_entropy(x, y.long() - 1, weight=None, ignore_index=-1, reduction='none')
    v = losses.mul_(weight).sum() / weight.sum()
    return v
