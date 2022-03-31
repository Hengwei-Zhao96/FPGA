#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/26 14:08
# @Author : Hw-Zhao
# @Site : 
# @File : base_dataset.py
# @Software: PyCharm

import numpy as np
from torch.utils.data import dataset
from util.image_pad import divisible_pad
from util.minibatch_sample import minibatch_sample

SEED = 2333


def fixed_num_sample(gt_mask: np.ndarray, num_train_samples, num_classes, seed=2333):
    rs = np.random.RandomState(seed)

    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    for i in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == i)[0]
        rs.shuffle(inds)

        train_inds = inds[:num_train_samples]
        test_inds = inds[num_train_samples:]

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1
    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)
    return train_indicator, test_indicator

class FullImageDataset(dataset.Dataset):
    def __init__(self, image, mask, train_flage, np_seed=2333, num_train_samples_per_class=200, sub_minibatch=10):
        self.image = image
        self.mask = mask
        self.train_flage = train_flage
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
   #     self.sub_minibatch = 1
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list length = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2147483648, size=9999)]
        self.preset()

    def preset(self):
        train_indicator, test_indicator = fixed_num_sample(self.mask, self.num_train_samples_per_class,
                                                           self.num_classes, self._seed)

        blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :]], axis=0)], 16)
        im = blob[0, :self.image.shape[-1], :, :]

        mask = blob[0, -3, :, :]
        self.train_indicator = blob[0, -2, :, :]
        self.test_indicator = blob[0, -1, :, :]

        if self.train_flage:
            self.train_inds_list = minibatch_sample(mask, self.train_indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.train_indicator, self.sub_minibatch,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 16

    def __getitem__(self, idx):
        if self.train_flage:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]
        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.train_flage:
            return len(self.train_inds_list)
        else:
            return 1
