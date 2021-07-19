import lmdb

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_opt, **kwargs):
        # dict to attr
        for kw, args in data_opt.items():
            setattr(self, kw, args)

        # can be used to override options defined in data_opt
        for kw, args in kwargs.items():
            setattr(self, kw, args)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def check_info(self, gt_keys, lr_keys):
        if len(gt_keys) != len(lr_keys):
            raise ValueError(
                f'GT & LR contain different numbers of images ({len(gt_keys)}  vs. {len(lr_keys)})')

        for i, (gt_key, lr_key) in enumerate(zip(gt_keys, lr_keys)):
            gt_info = self.parse_lmdb_key(gt_key)
            lr_info = self.parse_lmdb_key(lr_key)

            if gt_info[0] != lr_info[0]:
                raise ValueError(
                    f'video index mismatch ({gt_info[0]} vs. {lr_info[0]} for the {i} key)')

            gt_num, gt_h, gt_w = gt_info[1]
            lr_num, lr_h, lr_w = lr_info[1]
            s = self.scale
            if (gt_num != lr_num) or (gt_h != lr_h * s) or (gt_w != lr_w * s):
                raise ValueError(
                    f'video size mismatch ({gt_info[1]} vs. {lr_info[1]} for the {i} key)')

            if gt_info[2] != lr_info[2]:
                raise ValueError(
                    f'frame mismatch ({gt_info[2]} vs. {lr_info[2]} for the {i} key)')

    @staticmethod
    def init_lmdb(seq_dir):
        env = lmdb.open(
            seq_dir, readonly=True, lock=False, readahead=False, meminit=False)
        return env

    @staticmethod
    def parse_lmdb_key(key):
        key_lst = key.split('_')
        idx, size, frm = key_lst[:-2], key_lst[-2], int(key_lst[-1])
        idx = '_'.join(idx)
        size = tuple(map(int, size.split('x')))  # n_frm, h, w
        return idx, size, frm

    @staticmethod
    def read_lmdb_frame(env, key, size):
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        frm = np.frombuffer(buf, dtype=np.uint8).reshape(*size)
        return frm

    def crop_sequence(self, **kwargs):
        pass

    @staticmethod
    def augment_sequence(**kwargs):
        pass
