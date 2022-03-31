from scipy.io import loadmat
from data.base_data.base_dataset import FullImageDataset
import util.data_normalization as preprocess

SEED = 2333


class NewSalinasDataset(FullImageDataset):
    def __init__(self, config):
        self.im_mat_path = config['image_mat_path']
        self.gt_mat_path = config['gt_mat_path']

        im_mat = loadmat(self.im_mat_path)
        image = im_mat['salinas_corrected']
        gt_mat = loadmat(self.gt_mat_path)
        mask = gt_mat['salinas_gt']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        super(NewSalinasDataset, self).__init__(image, mask, config['training'], np_seed=SEED,
                                              num_train_samples_per_class=config['num_train_samples_per_class'],
                                              sub_minibatch=config['sub_minibatch'])
