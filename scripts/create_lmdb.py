import argparse
import glob
import os
import os.path as osp
import pickle
import random

import cv2
import lmdb
import numpy as np


def get_FileSize(filePath):

    fsize = os.path.getsize(filePath)
    return round(fsize, 3)


def create_lmdb(dataset, raw_dir, lmdb_dir, zip_img=False, filter_file=''):
    assert dataset in ('VimeoTecoGAN', 'VimeoTecoGAN-sub'), f'Unknown Dataset: {dataset}'
    print(f'Creating lmdb for dataset: {dataset}')

    # retrieve sequences
    if filter_file:  # dump selective data into lmdb
        with open(filter_file, 'r') as f:
            seq_dir_lst = sorted([line.strip() for line in f])
    else:
        seq_dir_lst = sorted(os.listdir(raw_dir))
    print(f'Number of sequences: {len(seq_dir_lst)}')

    # compute space to allocate
    print('Calculating space needed for LMDB generation ... ', end='')
    nbytes = 0
    for seq_dir in seq_dir_lst:
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_dir, '*.png')))
        if not zip_img:
            nbytes_per_frm = cv2.imread(frm_path_lst[0], cv2.IMREAD_UNCHANGED).nbytes
        else:
            nbytes_per_frm = get_FileSize(frm_path_lst[0])

        nbytes += len(frm_path_lst)*nbytes_per_frm
    alloc_size = round(1.5*nbytes)
    print(f'{alloc_size / (1 << 30):.2f} GB')

    # create lmdb environment
    env = lmdb.open(lmdb_dir, map_size=alloc_size)

    # write data to lmdb
    commit_freq = 5
    keys = []
    txn = env.begin(write=True)
    for b, seq_dir in enumerate(seq_dir_lst):
        # log
        print(f'Processing {seq_dir} ({b}/{len(seq_dir_lst)})\r', end='')

        # setup
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_dir, '*.png')))
        n_frm = len(frm_path_lst)

        # read frames
        for i in range(n_frm):
            frm = cv2.imread(frm_path_lst[i], cv2.IMREAD_UNCHANGED)
            frm = np.ascontiguousarray(frm[..., ::-1])  # hwc|rgb|uint8

            h, w, c = frm.shape
            key = f'{seq_dir}_{n_frm}x{h}x{w}_{i:04d}'
            if not zip_img:
                txn.put(key.encode('ascii'), frm)
            else:
                txn.put(key.encode('ascii'), cv2.imencode('.jpg', frm)[1])
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
        'keys': keys
    }
    pickle.dump(meta_info, open(osp.join(lmdb_dir, 'meta_info.pkl'), 'wb'))


def check_lmdb(dataset, lmdb_dir, zip_img=False, display=True):

    def visualize(win, img):
        if display:
            cv2.namedWindow(win, 0)
            cv2.resizeWindow(win, img.shape[-2], img.shape[-3])
            size = img.shape
            dst = cv2.resize(img, (size[1], size[0]), fx=0.5, fy=0.5)
            cv2.imshow(win, dst[..., ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite('_'.join(win.split('/')) + '.png', img[..., ::-1])

    assert dataset in ('VimeoTecoGAN', 'VimeoTecoGAN-sub'), f'Unknown Dataset: {dataset}'
    print(f'Checking lmdb dataset: {dataset}')

    # load keys
    meta_info = pickle.load(open(osp.join(lmdb_dir, 'meta_info.pkl'), 'rb'))
    keys = meta_info['keys']
    print(f'Number of keys: {len(keys)}')

    # randomly select frame to visualize
    with lmdb.open(lmdb_dir) as env:
        for i in range(3):  # replace to whatever you want
            idx = random.randint(0, len(keys) - 1)
            key = keys[idx]

            # parse key
            key_lst = key.split('_')
            vid, sz, frm = '_'.join(key_lst[:-2]), key_lst[-2], key_lst[-1]
            sz = tuple(map(int, sz.split('x')))
            sz = (*sz[1:], 3)
            print(f'video index: {vid} | size: {sz} | # of frame: {frm}')

            with env.begin() as txn:
                buf = txn.get(key.encode('ascii'))
                if not zip_img:
                    val = np.frombuffer(buf, dtype=np.uint8).reshape(*sz)  # HWC
                else:
                    val = np.frombuffer(buf, dtype=np.uint8)
                    val = cv2.imdecode(val, cv2.IMREAD_COLOR)

            visualize(key, val)


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='VimeoTecoGAN or VimeoTecoGAN-sub')
    parser.add_argument('--data_type', type=str, required=True,
                        help='GT or Bicubic4xLR')
    args = parser.parse_args()

    # setup
    raw_dir = f'data/{args.dataset}/{args.data_type}'
    lmdb_dir = f'data/{args.dataset}/{args.data_type}.lmdb'
    filter_file = ''

    # run
    if osp.exists(lmdb_dir):
        print(f'Dataset [{args.dataset}] already exists')
        print('Checking the LMDB dataset ...')
        check_lmdb(args.dataset, lmdb_dir, zip_img=True, display=False)
    else:
        create_lmdb(args.dataset, raw_dir, lmdb_dir, zip_img=True)

        print('Checking the LMDB dataset ...')
        check_lmdb(args.dataset, lmdb_dir, zip_img=True, display=False)
