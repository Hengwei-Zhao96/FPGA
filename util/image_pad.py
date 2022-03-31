import numpy as np
import torch

def divisible_pad(image_list, size_divisor=4):
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float32)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out