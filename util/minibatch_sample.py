import numpy as np

def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, minibatch_size, seed):
    rs = np.random.RandomState(seed)
    # split into N classes
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = dict()
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        rs.shuffle(inds)

        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            left = cnt * minibatch_size
            if left >= len(inds):
                continue
            # remain last batch though the real size is smaller than minibatch_size
            right = min((cnt + 1) * minibatch_size, len(inds))
            fetch_inds = inds[left:right]
            train_inds[fetch_inds] = 1
        cnt += 1
        if train_inds.sum() == 0:
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))
