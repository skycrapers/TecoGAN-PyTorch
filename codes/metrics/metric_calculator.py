import os
import os.path as osp
import json
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.distributed as dist

from codes.utils import base_utils, data_utils, net_utils
from codes.utils.dist_utils import master_only
from .LPIPS.models.dist_model import DistModel


class MetricCalculator():
    """ Metric calculator for model evaluation

        Currently supported metrics:
            * PSNR (RGB and Y)
            * LPIPS
            * tOF as described in TecoGAN paper

        TODO:
            * save/print metrics in a fixed order
    """

    def __init__(self, opt):
        # initialize
        self.metric_opt = opt['metric']
        self.device = torch.device(opt['device'])
        self.dist = opt['dist']
        self.rank = opt['rank']

        self.psnr_colorspace = ''
        self.dm = None

        self.reset()

        # update configs for each metric
        for metric_type, cfg in self.metric_opt.items():
            if metric_type.lower() == 'psnr':
                self.psnr_colorspace = cfg['colorspace']

            if metric_type.lower() == 'lpips':
                self.dm = DistModel()
                self.dm.initialize(
                    model=cfg['model'],
                    net=cfg['net'],
                    colorspace=cfg['colorspace'],
                    spatial=cfg['spatial'],
                    use_gpu=(opt['device'] == 'cuda'),
                    gpu_ids=[0 if not self.dist else opt['local_rank']],
                    version=cfg['version'])

    def reset(self):
        self.reset_per_sequence()
        self.metric_dict = OrderedDict()
        self.avg_metric_dict = OrderedDict()

    def reset_per_sequence(self):
        self.seq_idx_curr = ''
        self.true_img_cur = None
        self.pred_img_cur = None
        self.true_img_pre = None
        self.pred_img_pre = None

    def gather(self, seq_idx_lst):
        """ Gather results from all devices.
            Results will be updated into self.metric_dict on device 0
        """

        # mdict
        # {
        #     'seq_idx': {
        #         'metric1': [frm1, frm2, ...],
        #         'metric2': [frm1, frm2, ...]
        #     }
        # }
        mdict = self.metric_dict

        mtype_lst = self.metric_opt.keys()
        n_metric = len(mtype_lst)

        # avg_mdict
        # {
        #     'seq_idx': torch.tensor([metric1_avg, metric2_avg, ...])
        # }
        avg_mdict = {
            seq_idx: torch.zeros(n_metric, dtype=torch.float32, device=self.device)
            for seq_idx in seq_idx_lst
        }

        # average metric results for each sequence
        for i, mtype in enumerate(mtype_lst):
            for seq_idx, mdict_per_seq in mdict.items():  # ordered
                avg_mdict[seq_idx][i] += np.mean(mdict_per_seq[mtype])

        if self.dist:
            for seq_idx, tensor in avg_mdict.items():
                dist.reduce(tensor, dst=0)
            dist.barrier()

        if self.rank == 0:
            # avg_metric_dict
            # {
            #     'seq_idx': {
            #         'metric1': avg,
            #         'metric2': avg
            #     }
            # }

            for seq_idx in seq_idx_lst:
                self.avg_metric_dict[seq_idx] = OrderedDict([
                    (mtype, avg_mdict[seq_idx][i].item())
                    for i, mtype in enumerate(mtype_lst)
                ])

    def average(self):
        """ Return a dict including metric results averaged over all sequence
        """

        metric_avg_dict = OrderedDict()
        for mtype in self.metric_opt.keys():
            metric_all_seq = []
            for sqe_idx, mdict_per_seq in self.avg_metric_dict.items():
                metric_all_seq.append(mdict_per_seq[mtype])

            metric_avg_dict[mtype] = np.mean(metric_all_seq)

        return metric_avg_dict

    @master_only
    def display(self):
        # per sequence results
        for seq_idx, mdict_per_seq in self.avg_metric_dict.items():
            base_utils.log_info(f'Sequence: {seq_idx}')
            for mtype in self.metric_opt.keys():
                base_utils.log_info(f'\t{mtype}: {mdict_per_seq[mtype]:.6f}')

        # average results
        base_utils.log_info('Average')
        metric_avg_dict = self.average()
        for mtype, value in metric_avg_dict.items():
            base_utils.log_info(f'\t{mtype}: {value:.6f}')

    @master_only
    def save(self, model_idx, save_path, average=True, override=False):
        # load previous results if existed
        if osp.exists(save_path):
            with open(save_path, 'r') as f:
                json_dict = json.load(f)
        else:
            json_dict = dict()

        # update
        if model_idx not in json_dict:
            json_dict[model_idx] = OrderedDict()

        if average:
            metric_avg_dict = self.average()
            for mtype, value in metric_avg_dict.items():
                # override or skip
                if mtype in json_dict[model_idx] and not override:
                    continue
                json_dict[model_idx][mtype] = f'{value:.6f}'
        else:
            # TODO: save results of each sequence
            raise NotImplementedError()

        # sort
        json_dict = OrderedDict(sorted(
            json_dict.items(), key=lambda x: int(x[0].replace('G_iter', ''))))

        # save results
        with open(save_path, 'w') as f:
            json.dump(json_dict, f, sort_keys=False, indent=4)

    def compute_sequence_metrics(self, seq_idx, true_seq, pred_seq):
        # clear
        self.reset_per_sequence()

        # initialize metric_dict for the current sequence
        self.seq_idx_curr = seq_idx
        self.metric_dict[self.seq_idx_curr] = OrderedDict({
            metric: [] for metric in self.metric_opt.keys()})

        # compute metrics for each frame
        tot_frm = true_seq.shape[0]
        for i in range(tot_frm):
            self.true_img_cur = true_seq[i]  # hwc|rgb/y|uint8
            self.pred_img_cur = pred_seq[i]

            # pred_img and true_img may have different sizes
            # crop the larger one to match the smaller one
            true_h, true_w = self.true_img_cur.shape[:-1]
            pred_h, pred_w = self.pred_img_cur.shape[:-1]
            min_h, min_w = min(true_h, pred_h), min(true_w, pred_w)
            self.true_img_cur = self.true_img_cur[:min_h, :min_w, :]
            self.pred_img_cur = self.pred_img_cur[:min_h, :min_w, :]

            # compute metrics for the current frame
            self.compute_frame_metrics()

            # update
            self.true_img_pre = self.true_img_cur
            self.pred_img_pre = self.pred_img_cur

    def compute_frame_metrics(self):
        metric_dict = self.metric_dict[self.seq_idx_curr]

        # compute evaluation
        for metric_type, opt in self.metric_opt.items():
            if metric_type == 'PSNR':
                PSNR = self.compute_PSNR()
                metric_dict['PSNR'].append(PSNR)

            elif metric_type == 'LPIPS':
                LPIPS = self.compute_LPIPS()[0, 0, 0, 0].item()
                metric_dict['LPIPS'].append(LPIPS)

            elif metric_type == 'tOF':
                # skip the first frame
                if self.pred_img_pre is not None:
                    tOF = self.compute_tOF()
                    metric_dict['tOF'].append(tOF)

    def compute_PSNR(self):
        if self.psnr_colorspace == 'rgb':
            true_img = self.true_img_cur
            pred_img = self.pred_img_cur
        else:
            # convert to ycbcr, and keep the y channel
            true_img = data_utils.rgb_to_ycbcr(self.true_img_cur)[..., 0]
            pred_img = data_utils.rgb_to_ycbcr(self.pred_img_cur)[..., 0]

        diff = true_img.astype(np.float64) - pred_img.astype(np.float64)
        RMSE = np.sqrt(np.mean(np.power(diff, 2)))

        if RMSE == 0:
            return np.inf

        PSNR = 20 * np.log10(255.0 / RMSE)
        return PSNR

    def compute_LPIPS(self):
        true_img = np.ascontiguousarray(self.true_img_cur)
        pred_img = np.ascontiguousarray(self.pred_img_cur)

        # to tensor
        true_img = torch.FloatTensor(true_img).unsqueeze(0).permute(0, 3, 1, 2)
        pred_img = torch.FloatTensor(pred_img).unsqueeze(0).permute(0, 3, 1, 2)

        # normalize to [-1, 1]
        true_img = true_img.to(self.device) * 2.0 / 255.0 - 1.0
        pred_img = pred_img.to(self.device) * 2.0 / 255.0 - 1.0

        with torch.no_grad():
            LPIPS = self.dm.forward(true_img, pred_img)

        return LPIPS

    def compute_tOF(self):
        true_img_cur = cv2.cvtColor(self.true_img_cur, cv2.COLOR_RGB2GRAY)
        pred_img_cur = cv2.cvtColor(self.pred_img_cur, cv2.COLOR_RGB2GRAY)
        true_img_pre = cv2.cvtColor(self.true_img_pre, cv2.COLOR_RGB2GRAY)
        pred_img_pre = cv2.cvtColor(self.pred_img_pre, cv2.COLOR_RGB2GRAY)

        # forward flow
        true_OF = cv2.calcOpticalFlowFarneback(
            true_img_pre, true_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        pred_OF = cv2.calcOpticalFlowFarneback(
            pred_img_pre, pred_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # EPE
        diff_OF = true_OF - pred_OF
        tOF = np.mean(np.sqrt(np.sum(diff_OF**2, axis=-1)))

        return tOF
