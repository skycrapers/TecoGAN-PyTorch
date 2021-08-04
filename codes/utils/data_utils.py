import os
import os.path as osp

import cv2
import numpy as np
import torch
from scipy import signal


def create_kernel(sigma, ksize=None):
    if ksize is None:
        ksize = 1 + 2 * int(sigma * 3.0)

    gkern1d = signal.gaussian(ksize, std=sigma).reshape(ksize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gaussian_kernel = gkern2d / gkern2d.sum()
    zero_kernel = np.zeros_like(gaussian_kernel)

    kernel = np.float32([
        [gaussian_kernel, zero_kernel, zero_kernel],
        [zero_kernel, gaussian_kernel, zero_kernel],
        [zero_kernel, zero_kernel, gaussian_kernel]])

    kernel = torch.from_numpy(kernel)

    return kernel


def rgb_to_ycbcr(img):
    """ Coefficients are taken from the  official codes of DUF-VSR
        This conversion is also the same as that in BasicSR

        Parameters:
            :param  img: rgb image in type np.uint8
            :return: ycbcr image in type np.uint8
    """

    T = np.array([
        [0.256788235294118, -0.148223529411765, 0.439215686274510],
        [0.504129411764706, -0.290992156862745, -0.367788235294118],
        [0.097905882352941, 0.439215686274510, -0.071427450980392],
    ], dtype=np.float64)

    O = np.array([16, 128, 128], dtype=np.float64)

    img = img.astype(np.float64)
    res = np.matmul(img, T) + O
    res = res.clip(0, 255).round().astype(np.uint8)

    return res


def float32_to_uint8(inputs):
    """ Convert np.float32 array to np.uint8

        Parameters:
            :param input: np.float32, (NT)CHW, [0, 1]
            :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


def canonicalize(data):
    """ Convert data to torch float32 tensor

        Assume the input data has type np.uint8/np.float32 or torch.uint8/torch.float32,
        and uint8 data ranges in [0, 255] and float32 data ranges in [0, 1]
    """

    if isinstance(data, np.ndarray):
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        data = torch.from_numpy(np.ascontiguousarray(data))

    elif isinstance(data, torch.Tensor):
        if data.dtype == torch.uint8:
            data = data.float() / 255.0

    else:
        raise NotImplementedError()

    return data


def save_sequence(seq_dir, seq_data, frm_idx_lst=None, to_bgr=False):
    """ Save each frame of a sequence to .png image in seq_dir

        Parameters:
            :param seq_dir: dir to save results
            :param seq_data: sequence with shape thwc|uint8
            :param frm_idx_lst: specify filename for each frame to be saved
            :param to_bgr: whether to flip color channels
    """

    if to_bgr:
        seq_data = seq_data[..., ::-1]  # rgb2bgr

    # use default frm_idx_lst is not specified
    tot_frm = len(seq_data)
    if frm_idx_lst is None:
        frm_idx_lst = ['{:04d}.png'.format(i) for i in range(tot_frm)]

    # save for each frame
    os.makedirs(seq_dir, exist_ok=True)
    for i in range(tot_frm):
        cv2.imwrite(osp.join(seq_dir, frm_idx_lst[i]), seq_data[i])
