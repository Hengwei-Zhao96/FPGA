import numpy as np
from data.base_data.base_dataset import FullImageDataset
import util.data_normalization as preprocess
from util.image_pad import divisible_pad
from util.minibatch_sample import minibatch_sample
from osgeo import gdal

SEED = 2333

def read_ENVI(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        data = data.transpose((1, 2, 0))
    return data

class NewGRSS2013Dataset(FullImageDataset):
    def __init__(self, config):
        self.im_mat_path = config['image_mat_path']
        self.gt_mat_path = config['gt_mat_path']

        image = read_ENVI(self.im_mat_path)
        mask = read_ENVI(self.gt_mat_path)

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = config['training']
        self.sub_minibatch = config['sub_minibatch']
        super(NewGRSS2013Dataset, self).__init__(image, mask, config['training'], np_seed=SEED,
                                                 num_train_samples_per_class=None,
                                                 sub_minibatch=config['sub_minibatch'])

    def preset(self):
        indicator = np.where(self.mask != 0, np.ones_like(self.mask), np.zeros_like(self.mask))

        blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              indicator[None, :, :]], axis=0)], 16)
        im = blob[0, :self.image.shape[-1], :, :]

        mask = blob[0, -2, :, :]
        self.indicator = blob[0, -1, :, :]

        if self.training:
            self.train_inds_list = minibatch_sample(mask, self.indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.indicator, self.sub_minibatch,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 22

    def __getitem__(self, idx):

        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]

        else:
            return self.pad_im, self.pad_mask, self.indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1


