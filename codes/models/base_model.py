from collections import OrderedDict
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.data_utils import create_kernel, downsample_bd
from utils.dist_utils import master_only


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.scale = opt['scale']
        self.device = torch.device(opt['device'])
        self.blur_kernel = None
        self.dist = opt['dist']
        self.is_train = opt['is_train']

        if self.is_train:
            self.lr_data, self.gt_data = None, None
            self.ckpt_dir = opt['train']['ckpt_dir']
            self.log_decay = opt['logger'].get('decay', 0.99)
            self.log_dict = OrderedDict()
            self.running_log_dict = OrderedDict()

    def set_networks(self):
        pass

    def set_criterions(self):
        pass

    def set_optimizers(self):
        pass

    def set_lr_schedules(self):
        pass

    def prepare_training_data(self, data):
        """ prepare gt, lr data for training

            for BD degradation, generate lr data and remove the border of gt data
            for BI degradation, use input data directly
        """

        degradation_type = self.opt['dataset']['degradation']['type']

        if degradation_type == 'BI':
            self.gt_data = data['gt'].to(self.device)
            self.lr_data = data['lr'].to(self.device)

        elif degradation_type == 'BD':
            # generate lr data on the fly (on gpu)

            # set params
            scale = self.opt['scale']
            sigma = self.opt['dataset']['degradation'].get('sigma', 1.5)
            border_size = int(sigma * 3.0)

            gt_data = data['gt'].to(self.device)  # with border
            n, t, c, gt_h, gt_w = gt_data.size()
            lr_h = (gt_h - 2*border_size)//scale
            lr_w = (gt_w - 2*border_size)//scale

            # create blurring kernel
            if self.blur_kernel is None:
                self.blur_kernel = create_kernel(sigma).to(self.device)
            blur_kernel = self.blur_kernel

            # generate lr data
            gt_data = gt_data.view(n*t, c, gt_h, gt_w)
            lr_data = downsample_bd(gt_data, blur_kernel, scale, pad_data=False)
            lr_data = lr_data.view(n, t, c, lr_h, lr_w)

            # remove gt border
            gt_data = gt_data[
                ...,
                border_size: border_size + scale*lr_h,
                border_size: border_size + scale*lr_w]
            gt_data = gt_data.view(n, t, c, scale*lr_h, scale*lr_w)

            self.gt_data, self.lr_data = gt_data, lr_data  # tchw|float32

    def prepare_inference_data(self, data):
        """ Prepare lr data for training (w/o loading on device)
        """

        degradation_type = self.opt['dataset']['degradation']['type']

        if degradation_type == 'BI':
            self.lr_data = data['lr']

        elif degradation_type == 'BD':
            if 'lr' in data:
                self.lr_data = data['lr']
            else:
                # generate lr data on the fly (on cpu)
                # TODO: do frame-wise downsampling on gpu for acceleration?
                gt_data = data['gt']  # thwc|uint8

                # set params
                scale = self.opt['scale']
                sigma = self.opt['dataset']['degradation'].get('sigma', 1.5)

                # create blurring kernel
                if self.blur_kernel is None:
                    self.blur_kernel = create_kernel(sigma)
                blur_kernel = self.blur_kernel.cpu()

                # generate lr data
                gt_data = gt_data.permute(0, 3, 1, 2).float()  / 255.0  # tchw|float32
                lr_data = downsample_bd(
                    gt_data, blur_kernel, scale, pad_data=True)
                lr_data = lr_data.permute(0, 2, 3, 1)  # thwc|float32

                self.lr_data = lr_data

        # thwc to tchw
        self.lr_data = self.lr_data.permute(0, 3, 1, 2)  # tchw|float32

    def train(self):
        pass

    def infer(self):
        pass

    def model_to_device(self, net):
        net = net.to(self.device)
        if self.dist:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()])
        return net

    def update_learning_rate(self):
        if hasattr(self, 'sched_G') and self.sched_G is not None:
            self.sched_G.step()

        if hasattr(self, 'sched_D') and self.sched_D is not None:
            self.sched_D.step()

    def get_learning_rate(self):
        lr_dict = OrderedDict()

        if hasattr(self, 'optim_G'):
            lr_dict['lr_G'] = self.optim_G.param_groups[0]['lr']

        if hasattr(self, 'optim_D'):
            lr_dict['lr_D'] = self.optim_D.param_groups[0]['lr']

        return lr_dict

    def reduce_log(self):
        if self.dist:
            rank, world_size = self.opt['rank'], self.opt['world_size']
            with torch.no_grad():
                keys, vals = [], []
                for key, val in self.log_dict.items():
                    keys.append(key)
                    vals.append(val)
                vals = torch.FloatTensor(vals).to(self.device)
                dist.reduce(vals, dst=0)
                if rank == 0:  # average
                    vals /= world_size
                self.log_dict = {key: val.item() for key, val in zip(keys, vals)}

    def update_running_log(self):
        self.reduce_log()  # for distributed training

        d = self.log_decay
        for k in self.log_dict.keys():
            current_val = self.log_dict[k]
            running_val = self.running_log_dict.get(k)

            if running_val is None:
                running_val = current_val
            else:
                running_val = d * running_val + (1.0 - d) * current_val

            self.running_log_dict[k] = running_val

    def get_current_log(self):
        return self.log_dict

    def get_running_log(self):
        return self.running_log_dict

    def get_format_msg(self, epoch, iter):
        # generic info
        msg = f'[epoch: {epoch} | iter: {iter}'
        for lr_type, lr in self.get_learning_rate().items():
            msg += f' | {lr_type}: {lr:.2e}'
        msg += '] '

        # loss info
        log_dict = self.get_running_log()
        msg += ', '.join([f'{k}: {v:.3e}' for k, v in log_dict.items()])

        return msg

    def save(self, current_iter):
        pass

    @staticmethod
    def get_bare_model(net):
        if isinstance(net, DistributedDataParallel):
            net = net.module
        return net

    @master_only
    def save_network(self, net, net_label, current_iter):
        filename = f'{net_label}_iter{current_iter}.pth'
        save_path = osp.join(self.ckpt_dir, filename)
        net = self.get_bare_model(net)
        torch.save(net.state_dict(), save_path)

    def save_training_state(self, current_epoch, current_iter):
        # TODO
        pass

    def load_network(self, net, load_path):
        state_dict = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        net = self.get_bare_model(net)
        net.load_state_dict(state_dict)

    def pad_sequence(self, lr_data):
        """
            Parameters:
                :param lr_data: tensor in shape tchw
        """
        padding_mode = self.opt['test'].get('padding_mode', 'reflect')
        n_pad_front = self.opt['test'].get('num_pad_front', 0)
        assert n_pad_front < lr_data.size(0)

        # pad
        if padding_mode == 'reflect':
            lr_data = torch.cat(
                [lr_data[1: 1 + n_pad_front, ...].flip(0), lr_data], dim=0)

        elif padding_mode == 'replicate':
            lr_data = torch.cat(
                [lr_data[:1, ...].expand(n_pad_front, -1, -1, -1), lr_data], dim=0)

        else:
            raise ValueError(f'Unrecognized padding mode: {padding_mode}')

        return lr_data, n_pad_front
