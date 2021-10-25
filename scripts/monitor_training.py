import os.path as osp
import argparse
import json
import math
import re
import bisect

import matplotlib.pyplot as plt


# -------------------- utility functions -------------------- #
def append(loss_dict, loss_name, loss_value):
    if loss_name not in loss_dict:
        loss_dict[loss_name] = [loss_value]
    else:
        loss_dict[loss_name].append(loss_value)


def split(pattern, string):
    return re.split(r'\s*{}\s*'.format(pattern), string)


def parse_log(log_file):
    # define loss patterns
    loss_pattern = r'.*\[epoch:.*\| iter: (\d+).*\] (.*)'

    # load log file
    with open(log_file, 'r') as f:
        lines = [line.strip() for line in f]

    # parse log file
    loss_dict = {}    # {'iter': [], 'loss1': [], 'loss2':[], ...}
    for line in lines:
        loss_match = re.match(loss_pattern, line)
        if loss_match:
            iter = int(loss_match.group(1))
            append(loss_dict, 'iter', iter)
            for s in split(',', loss_match.group(2)):
                if s:
                    k, v = split(':', s)
                    append(loss_dict, k, float(v))

    return loss_dict


def parse_json(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    metric_dict = {}
    for model_idx, metrics in json_dict.items():
        append(metric_dict, 'iter', int(model_idx.replace('G_iter', '')))
        for metric, val in metrics.items():
            append(metric_dict, metric, float(val))

    return metric_dict


def plot_curve(ax, iter, value, style='-', alpha=1.0, label='',
               start_iter=0, end_iter=-1, smooth=0, linewidth=1.0):

    assert len(iter) == len(value), \
        'mismatch in <iter> and <value> ({} vs {})'.format(
            len(iter), len(value))
    l = len(iter)

    if smooth:
        for i in range(1, l):
            value[i] = smooth * value[i - 1] + (1 - smooth) * value[i]

    start_index = bisect.bisect_left(iter, start_iter)
    end_index = l if end_iter < 0 else bisect.bisect_right(iter, end_iter)
    ax.plot(
        iter[start_index:end_index], value[start_index:end_index],
        style, alpha=alpha, label=label, linewidth=linewidth)


def plot_loss_curves(loss_dict, ax, loss_type, start_iter=0, end_iter=-1,
                     smooth=0):

    for model_idx, model_loss_dict in loss_dict.items():
        if loss_type in model_loss_dict:
            plot_curve(
                ax, model_loss_dict['iter'], model_loss_dict[loss_type],
                alpha=1.0, label=model_idx, start_iter=start_iter,
                end_iter=end_iter, smooth=smooth)
    ax.legend(loc='best', fontsize='small')
    ax.set_ylabel(loss_type)
    ax.set_xlabel('iteration')
    plt.grid(True)


def plot_metric_curves(metric_dict, ax, metric_type, start_iter=0, end_iter=-1):
    """ currently can only plot average results
    """

    for model_idx, model_metric_dict in metric_dict.items():
        if metric_type in model_metric_dict:
            plot_curve(
                ax, model_metric_dict['iter'], model_metric_dict[metric_type],
                alpha=1.0, label=model_idx, start_iter=start_iter,
                end_iter=end_iter)
    ax.legend(loc='best', fontsize='small')
    ax.set_ylabel(metric_type)
    ax.set_xlabel('iteration')
    plt.grid(True)


# -------------------- monitor -------------------- #
def monitor(root_dir, testset, exp_id_lst, loss_lst, metric_lst):
    # ================ basic settings ================#
    start_iter = 0
    loss_smooth = 0

    # ================ parse logs ================#
    loss_dict = {}    # {'model1': {'loss1': x1, ...}, ...}
    metric_dict = {}  # {'model1': {'metric1': x1, ...}, ...}
    for exp_id in exp_id_lst:
        # parse log
        log_file = osp.join(root_dir, exp_id, 'train', 'train.log')
        if osp.exists(log_file):
            loss_dict[exp_id] = parse_log(log_file)

        # parse json
        json_file = osp.join(
            root_dir, exp_id, 'test', 'metrics', f'{testset}_avg.json')
        if osp.exists(json_file):
            metric_dict[exp_id] = parse_json(json_file)

    # ================ plot loss curves ================#
    n_loss = len(loss_lst)
    base_figsize = (12, 2 * math.ceil(n_loss / 2))
    fig = plt.figure(1, figsize=base_figsize)
    for i in range(n_loss):
        ax = fig.add_subplot('{}{}{}'.format(math.ceil(n_loss / 2), 2, i + 1))
        plot_loss_curves(
            loss_dict, ax, loss_lst[i], start_iter=start_iter, smooth=loss_smooth)

    # ================ plot metric curves ================#
    n_metric = len(metric_lst)
    base_figsize = (12, 2 * math.ceil(n_metric / 2))
    fig = plt.figure(2, figsize=base_figsize)
    for i in range(n_metric):
        ax = fig.add_subplot('{}{}{}'.format(math.ceil(n_metric / 2), 2, i + 1))
        plot_metric_curves(
            metric_dict, ax, metric_lst[i], start_iter=start_iter)

    plt.show()


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--degradation', '-dg', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--dataset', '-ds', type=str, required=True)
    args = parser.parse_args()

    # select model
    root_dir = '.'
    if 'FRVSR' in args.model:
        # select experiments
        exp_id_lst = [
            # experiment dirs
            f'experiments_{args.degradation}/{args.model}',
        ]

        # select losses
        loss_lst = [
            'l_pix_G',  # pixel loss
            'l_warp_G',  # warping loss
        ]

        # select metrics
        metric_lst = [
            'PSNR',
        ]

    elif 'TecoGAN' in args.model:
        # select experiments
        exp_id_lst = [
            # experiment dirs
            f'experiments_{args.degradation}/{args.model}',
        ]

        # select losses
        loss_lst = [
            'l_pix_G',   # pixel loss
            'l_warp_G',  # warping loss
            'l_feat_G',  # perceptual loss
            'l_gan_G',   # generator loss
            'l_gan_D',   # discriminator loss
            'p_real_D',
            'p_fake_D',
        ]

        # select metrics
        metric_lst = [
            'PSNR',
            'LPIPS',
            'tOF'
        ]

    else:
        raise ValueError(f'Unrecoginzed model: {args.model}')

    monitor(root_dir, args.dataset, exp_id_lst, loss_lst, metric_lst)
