#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/26 17:22
# @Author : Hw-Zhao
# @Site : 
# @File : data_normalization.py
# @Software: PyCharm
import numpy as np
import torch

def _np_mean_std_normalize(image, mean, std):
    """
    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray
    Returns:
    """
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean, np.float32)
    if not isinstance(std, np.ndarray):
        std = np.array(std, np.float32)
    shape = [1] * image.ndim
    shape[-1] = -1
    return (image - mean.reshape(shape)) / std.reshape(shape)


def _th_mean_std_normalize(image, mean, std):
    """ this version faster than torchvision.transforms.functional.normalize
    Args:
        image: 3-D or 4-D array of shape [batch (optional) , height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray
    Returns:
    """
    shape = [1] * image.dim()
    shape[-1] = -1
    mean = torch.tensor(mean, requires_grad=False).reshape(*shape)
    std = torch.tensor(std, requires_grad=False).reshape(*shape)

    return image.sub(mean).div(std)


def mean_std_normalize(image, mean, std):
    """
    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple
        std: a list or tuple
    Returns:
    """
    if isinstance(image, np.ndarray):
        return _np_mean_std_normalize(image, mean, std)
    elif isinstance(image, torch.Tensor):
        return _th_mean_std_normalize(image, mean, std)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))
