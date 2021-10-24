import os
import os.path as osp
import argparse
import glob
import lmdb
import pickle
import random

import numpy as np
import cv2


def create_lmdb(dataset, raw_dir, lmdb_dir, filter_file=''):
    assert dataset in ('VimeoTecoGAN', 'REDS'), f'Unknown Dataset: {dataset}'
    print(f'>> Start to create lmdb for {dataset}')

    # scan dir
    if filter_file:  # use sequences specified by the filter_file
        with open(filter_file, 'r') as f:
            seq_idx_lst = sorted([line.strip() for line in f])
    else:  # use all found sequences
        seq_idx_lst = sorted(os.listdir(raw_dir))

    num_seq = len(seq_idx_lst)
    print(f'>> Number of sequences: {num_seq}')

    # compute space to be allocated
    nbytes = 0
    for seq_idx in seq_idx_lst:
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_idx, '*.png')))
        nbytes_per_frm = cv2.imread(frm_path_lst[0], cv2.IMREAD_UNCHANGED).nbytes
        nbytes += len(frm_path_lst) * nbytes_per_frm
    alloc_size = round(2 * nbytes)
    print(f'>> Space required for lmdb generation: {alloc_size / (1 << 30):.2f} GB')

    # create lmdb environment
    env = lmdb.open(lmdb_dir, map_size=alloc_size)

    # write data to lmdb
    commit_freq = 5
    keys = []
    txn = env.begin(write=True)
    for b, seq_idx in enumerate(seq_idx_lst):
        # log
        print(f'   Processing sequence: {seq_idx} ({b + 1}/{num_seq})\r', end='')

        # get info
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_idx, '*.png')))
        n_frm = len(frm_path_lst)

        # read frames
        for i in range(n_frm):
            frm = cv2.imread(frm_path_lst[i], cv2.IMREAD_UNCHANGED)
            frm = np.ascontiguousarray(frm[..., ::-1])  # hwc|rgb|uint8

            h, w, c = frm.shape
            key = f'{seq_idx}_{n_frm}x{h}x{w}_{i:04d}'

            txn.put(key.encode('ascii'), frm)
            keys.append(key)

        # commit
        if b % commit_freq == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    # create meta information
    meta_info = {
        'name': dataset,
        'color': 'RGB',
        'keys': keys
    }
    pickle.dump(meta_info, open(osp.join(lmdb_dir, 'meta_info.pkl'), 'wb'))

    print(f'>> Finished lmdb generation for {dataset}')


def check_lmdb(dataset, lmdb_dir):

    def visualize(win, img):
        cv2.namedWindow(win, 0)
        cv2.resizeWindow(win, img.shape[-2], img.shape[-3])
        cv2.imshow(win, img[..., ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    assert dataset in ('VimeoTecoGAN', 'REDS'), f'Unknown Dataset: {dataset}'
    print(f'>> Start to check lmdb dataset: {dataset}.lmdb')

    # load keys
    meta_info = pickle.load(open(osp.join(lmdb_dir, 'meta_info.pkl'), 'rb'))
    keys = meta_info['keys']
    print(f'>> Number of keys: {len(keys)}')

    # randomly select frames for visualization
    with lmdb.open(lmdb_dir) as env:
        for i in range(3):  # can be replaced to any number
            idx = random.randint(0, len(keys) - 1)
            key = keys[idx]

            # parse key
            key_lst = key.split('_')
            vid, sz, frm = '_'.join(key_lst[:-2]), key_lst[-2], key_lst[-1]
            sz = tuple(map(int, sz.split('x')))
            sz = (*sz[1:], 3)
            print(f'   Visualizing frame: #{frm} from sequence: {vid} (size: {sz})')

            with env.begin() as txn:
                buf = txn.get(key.encode('ascii'))
                val = np.frombuffer(buf, dtype=np.uint8).reshape(*sz) # hwc

            visualize(key, val)

    print(f'>> Finished lmdb checking for {dataset}')


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='VimeoTecoGAN | REDS')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Dir to the raw data')
    parser.add_argument('--lmdb_dir', type=str, required=True,
                        help='Dir to the lmdb data')
    parser.add_argument('--filter_file', type=str, default='',
                        help='File used to select sequences')
    args = parser.parse_args()

    # run
    if osp.exists(args.lmdb_dir):
        print(f'>> Dataset [{args.dataset}] already exists.')
        check_lmdb(args.dataset, args.lmdb_dir)
    else:
        create_lmdb(args.dataset, args.raw_dir, args.lmdb_dir, args.filter_file)
        check_lmdb(args.dataset, args.lmdb_dir)
