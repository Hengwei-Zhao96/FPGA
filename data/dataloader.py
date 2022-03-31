#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/26 17:12
# @Author : Hw-Zhao
# @Site : 
# @File : dataloader.py
# @Software: PyCharm
from torch.utils.data.dataloader import DataLoader
from data.base_data.base_sampler import MinibatchSampler
from data.pavia_dataset import NewPaviaDataset
from data.igrss2013_dataset import NewGRSS2013Dataset
from data.salinas_dataset import NewSalinasDataset


class NewPaviaLoader(DataLoader):
    def __init__(self, config):
        self.config = config

        dataset = NewPaviaDataset(self.config)
        sampler = MinibatchSampler(dataset)
        super(NewPaviaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.config['num_workers'],
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

class NewSalinasLoader(DataLoader):
    def __init__(self, config):
        self.config = config

        dataset = NewSalinasDataset(self.config)
        sampler = MinibatchSampler(dataset)
        super(NewSalinasLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.config['num_workers'],
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)


class NewGrss2013Loader(DataLoader):
    def __init__(self, config):
        self.config = config

        dataset = NewGRSS2013Dataset(self.config)
        sampler = MinibatchSampler(dataset)

        super(NewGrss2013Loader, self).__init__(dataset,
                                                batch_size=1,
                                                sampler=sampler,
                                                batch_sampler=None,
                                                num_workers=self.config['num_workers'],
                                                pin_memory=True,
                                                drop_last=False,
                                                timeout=0,
                                                worker_init_fn=None)
