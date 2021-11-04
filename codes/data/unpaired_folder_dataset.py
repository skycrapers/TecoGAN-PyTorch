import os
import os.path as osp

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.base_utils import retrieve_files


class UnpairedFolderDataset(BaseDataset):
    """ Folder dataset for unpaired data (for BD degradation)
    """

    def __init__(self, data_opt, **kwargs):
        super(UnpairedFolderDataset, self).__init__(data_opt, **kwargs)

        # get keys
        self.keys = sorted(os.listdir(self.gt_seq_dir))

        # filter keys
        sel_keys = set(self.keys)
        if hasattr(self, 'filter_file') and self.filter_file is not None:
            with open(self.filter_file, 'r') as f:
                sel_keys = {line.strip() for line in f}
        elif hasattr(self, 'filter_list') and self.filter_list is not None:
            sel_keys = set(self.filter_list)
        self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        # load gt frames
        gt_seq = []
        for frm_path in retrieve_files(osp.join(self.gt_seq_dir, key)):
            gt_frm = cv2.imread(frm_path)[..., ::-1]
            gt_seq.append(gt_frm)
        gt_seq = np.stack(gt_seq)  # thwc|rgb|uint8

        # convert to tensor
        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq))  # uint8

        # gt: thwc|rgb|uint8
        return {
            'gt': gt_tsr,
            'seq_idx': key,
            'frm_idx': sorted(os.listdir(osp.join(self.gt_seq_dir, key)))
        }
