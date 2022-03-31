import torch
from torch.utils import data
from data.base_data.base_dataset import FullImageDataset

SEED = 2333


class MinibatchSampler(data.Sampler):
    def __init__(self, dataset: FullImageDataset):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(SEED)

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)
