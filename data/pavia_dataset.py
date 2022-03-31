#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/26 17:19
# @Author : Hw-Zhao
# @Site : 
# @File : pavia_dataset.py
# @Software: PyCharm

from scipy.io import loadmat
from data.base_data.base_dataset import FullImageDataset
import util.data_normalization as preprocess

SEED = 2333


class NewPaviaDataset(FullImageDataset):
    def __init__(self, config):
        self.im_mat_path = config['image_mat_path']
        self.gt_mat_path = config['gt_mat_path']

        im_mat = loadmat(self.im_mat_path)
        image = im_mat['paviaU']
        gt_mat = loadmat(self.gt_mat_path)
        mask = gt_mat['paviaU_gt']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        super(NewPaviaDataset, self).__init__(image, mask, config['training'], np_seed=SEED,
                                              num_train_samples_per_class=config['num_train_samples_per_class'],
                                              sub_minibatch=config['sub_minibatch'])
