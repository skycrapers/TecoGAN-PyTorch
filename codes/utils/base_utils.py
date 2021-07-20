import os
import os.path as osp
import random
import logging
import argparse
import yaml

import numpy as np
import torch

from .dist_utils import init_dist, master_only


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory of the current experiment')
    parser.add_argument('--mode', type=str, required=True,
                        help='which mode to use (train|test|profile)')
    parser.add_argument('--opt', type=str, required=True,
                        help='path to the config yaml file')
    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help='GPU index (set -1 to use CPU)')
    parser.add_argument('--lr_size', type=str, default='3x256x256',
                        help='size of the input frame')
    parser.add_argument('--test_speed', action='store_true',
                        help='whether to test speed')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local gpu index')
    return parser.parse_args()


def parse_configs(args):
    # load option file
    with open(osp.join(args.exp_dir, args.opt), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    opt['exp_dir'] = args.exp_dir
    opt['gpu_ids'] = args.gpu_ids
    opt['is_train'] = (args.mode == 'train')

    # setup device
    setup_device(opt, args.gpu_ids, args.local_rank)

    # setup random seed
    setup_random_seed(opt.get('manual_seed', 2021) + opt['rank'])

    return opt


def setup_device(opt, gpu_ids, local_rank):
    gpu_ids = tuple(map(int, gpu_ids.split(',')))
    if gpu_ids[0] < 0 or not torch.cuda.is_available():
        # cpu settings
        opt.update({
            'dist': False,
            'device': 'cpu',
            'rank': 0
        })
    else:
        # gpu settings
        if len(gpu_ids) == 1:
            # single gpu
            torch.cuda.set_device(0)
            opt.update({
                'dist': False,
                'device': 'cuda',
                'rank': 0
            })
        else:
            # multiple gpus
            init_dist(opt, local_rank)

        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name):
    # create a logger
    base_logger = logging.getLogger(name=name)
    base_logger.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s')
    # create a stream handler & set format
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # add handlers
    base_logger.addHandler(sh)


@master_only
def log_info(msg, logger_name='base'):
    logger = logging.getLogger(logger_name)
    logger.info(msg)


def print_options(opt, logger_name='base', tab=''):
    for key, val in opt.items():
        if isinstance(val, dict):
            log_info('{}{}:'.format(tab, key), logger_name)
            print_options(val, logger_name, tab + '  ')
        else:
            log_info('{}{}: {}'.format(tab, key, val), logger_name)


def retrieve_files(dir, suffix='png|jpg'):
    """ retrive files with specific suffix under dir and sub-dirs recursively
    """

    def retrieve_files_recursively(dir, file_lst):
        for d in sorted(os.listdir(dir)):
            dd = osp.join(dir, d)

            if osp.isdir(dd):
                retrieve_files_recursively(dd, file_lst)
            else:
                if osp.splitext(d)[-1].lower() in ['.' + s for s in suffix]:
                    file_lst.append(dd)

    if not dir:
        return []

    if isinstance(suffix, str):
        suffix = suffix.split('|')

    file_lst = []
    retrieve_files_recursively(dir, file_lst)
    file_lst.sort()

    return file_lst


def setup_paths(opt, mode):

    def setup_ckpt_dir():
        ckpt_dir = opt['train'].get('ckpt_dir', '')
        if not ckpt_dir:  # default dir
            ckpt_dir = osp.join(opt['exp_dir'], 'train', 'ckpt')
            opt['train']['ckpt_dir'] = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

    def setup_res_dir():
        res_dir = opt['test'].get('res_dir', '')
        if not res_dir:  # default dir
            res_dir = osp.join(opt['exp_dir'], 'test', 'results')
            opt['test']['res_dir'] = res_dir
        os.makedirs(res_dir, exist_ok=True)

    def setup_json_path():
        json_dir = opt['test'].get('json_dir', '')
        if not json_dir:  # default dir
            json_dir = osp.join(opt['exp_dir'], 'test', 'metrics')
            opt['test']['json_dir'] = json_dir
        os.makedirs(json_dir, exist_ok=True)

    def setup_model_path():
        load_path = opt['model']['generator'].get('load_path', '')
        if not load_path:
            raise ValueError(
                'Pretrained generator model is needed for testing')

        # parse models
        ckpt_dir, model_idx = osp.split(load_path)
        model_idx = osp.splitext(model_idx)[0]
        if model_idx == '*':
            # test a serial of models  TODO: check validity
            start_iter = opt['test']['start_iter']
            end_iter = opt['test']['end_iter']
            freq = opt['test']['test_freq']
            opt['model']['generator']['load_path_lst'] = [
                osp.join(ckpt_dir, f'G_iter{iter}.pth')
                for iter in range(start_iter, end_iter + 1, freq)]
        else:
            # test a single model
            opt['model']['generator']['load_path_lst'] = [
                osp.join(ckpt_dir, f'{model_idx}.pth')]

    if mode == 'train':
        setup_ckpt_dir()

        # for validation purpose
        for dataset_idx in opt['dataset'].keys():
            if 'test' not in dataset_idx:
                continue

            if opt['test'].get('save_res', False):
                setup_res_dir()

            if opt['test'].get('save_json', False):
                setup_json_path()

    elif mode == 'test':
        setup_model_path()

        for dataset_idx in opt['dataset'].keys():
            if 'test' not in dataset_idx:
                continue

            if opt['test'].get('save_res', False):
                setup_res_dir()

            if opt['test'].get('save_json', False):
                setup_json_path()
