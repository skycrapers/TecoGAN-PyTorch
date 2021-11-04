import os
import os.path as osp
import glob
from multiprocessing import Pool

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from codes.utils.data_utils import float32_to_uint8


""" Note: 
    There are two implementations (opencv's GaussianBlur and PyTorch's conv2d)
    for generating BD degraded LR data. Although there's only a slight numerical
    difference between these two methods, it's recommended to use the later one
    since it is adopted in model training.
"""

# setup params
# default settings
scale = 4  # downsampling scale
sigma = 1.5
ksize = 1 + 2 * int(sigma * 3.0)
# to be modified
n_process = 16  # the number of process to be used for downsampling
filepaths = glob.glob('../data/Vid4/GT/*/*.png')
gt_dir_idx, lr_dir_idx = 'GT', 'Gaussian4xLR'


def down_opencv(img, sigma, ksize, scale):
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)  # hwc|uint8
    lr_img = blur_img[::scale, ::scale].astype(np.float32) / 255.0
    return lr_img  # hwc|float32


def down_pytorch(img, sigma, ksize, scale):
    img = np.ascontiguousarray(img)
    img = torch.FloatTensor(img).unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # nchw

    gaussian_filters = create_kernel(sigma, ksize)

    filters_h, filters_w = gaussian_filters.shape[-2:]
    pad_h, pad_w = filters_h - 1, filters_w - 1

    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l

    img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), 'reflect')

    lr_img = F.conv2d(
        img, gaussian_filters, stride=scale, bias=None, padding=0)

    return lr_img[0].permute(1, 2, 0).numpy()  # hwc|float32


def downsample_worker(filepath):
    # log
    print('Processing {}'.format(filepath))

    # setup dirs
    gt_folder, img_idx = osp.split(filepath)
    lr_folder = gt_folder.replace(gt_dir_idx, lr_dir_idx)

    # read image
    img = cv2.imread(filepath)  # hwc|bgr|uint8

    # dowmsample
    # img_lr = down_opencv(img, sigma, ksize, scale)
    img_lr = down_pytorch(img, sigma, ksize, scale)
    img_lr = float32_to_uint8(img_lr)

    # save image
    cv2.imwrite(osp.join(lr_folder, img_idx), img_lr)


if __name__ == '__main__':
    # setup dirs
    print('# of images: {}'.format(len(filepaths)))
    for filepath in filepaths:
        gt_folder, _ = osp.split(filepath)
        lr_folder = gt_folder.replace(gt_dir_idx, lr_dir_idx)
        if not osp.exists(lr_folder):
            os.makedirs(lr_folder)

    # for each image
    pool = Pool(n_process)
    for filepath in sorted(filepaths):
        pool.apply_async(downsample_worker, args=(filepath,))
    pool.close()
    pool.join()
