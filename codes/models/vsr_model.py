from collections import OrderedDict

import torch.optim as optim
from utils import base_utils, data_utils, net_utils

from .base_model import BaseModel
from .networks import define_generator
from .optim import define_criterion, define_lr_schedule


class VSRModel(BaseModel):
    """ A model wrapper for objective video super-resolution
    """

    def __init__(self, opt):
        super(VSRModel, self).__init__(opt)

        # define network
        self.set_networks()

        # config training
        if self.is_train:
            self.set_criterions()
            self.set_optimizers()
            self.set_lr_schedules()

    def set_networks(self):
        # define generator
        self.net_G = define_generator(self.opt)
        self.net_G = self.model_to_device(self.net_G)
        base_utils.log_info('Generator: {}\n{}'.format(
            self.opt['model']['generator']['name'], self.net_G.__str__()))

        # load generator
        load_path_G = self.opt['model']['generator'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            base_utils.log_info('Load generator from: {}'.format(load_path_G))

    def set_criterions(self):
        # pixel criterion
        self.pix_crit = define_criterion(
            self.opt['train'].get('pixel_crit'))

        # warping criterion
        self.warp_crit = define_criterion(
            self.opt['train'].get('warping_crit'))

    def set_optimizers(self):
        self.optim_G = optim.Adam(
            self.net_G.parameters(),
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=self.opt['train']['generator'].get('betas', (0.9, 0.999)))

    def set_lr_schedules(self):
        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

    def train(self):
        # === initialize === #
        self.net_G.train()
        self.optim_G.zero_grad()

        # === forward net_G === #
        net_G_output_dict = self.net_G(self.lr_data)
        hr_data = net_G_output_dict['hr_data']

        # === optimize net_G === #
        loss_G = 0
        self.log_dict = OrderedDict()

        # pixel loss
        pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
        loss_pix_G = pix_w * self.pix_crit(hr_data, self.gt_data)
        loss_G += loss_pix_G
        self.log_dict['l_pix_G'] = loss_pix_G.item()

        # warping loss
        if self.warp_crit is not None:
            # warp lr_prev according to lr_flow
            lr_curr = net_G_output_dict['lr_curr']
            lr_prev = net_G_output_dict['lr_prev']
            lr_flow = net_G_output_dict['lr_flow']
            lr_warp = net_utils.backward_warp(lr_prev, lr_flow)

            warp_w = self.opt['train']['warping_crit'].get('weight', 1)
            loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr)
            loss_G += loss_warp_G
            self.log_dict['l_warp_G'] = loss_warp_G.item()

        # optimize
        loss_G.backward()
        self.optim_G.step()

    def infer(self, lr_data):
        """
            Parameters:
                :param lr_data: a rgb video sequence with shape thwc
                :return: video sequence (type np.uint8 and shape thwc)
        """

        # canonicalize
        lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor
        lr_data = lr_data.permute(0, 3, 1, 2)  # tchw

        # temporal padding
        lr_data, n_pad_front = self.pad_sequence(lr_data)

        # infer
        self.net_G.eval()
        hr_seq = self.net_G(lr_data, self.device)
        hr_seq = hr_seq[n_pad_front:]

        return hr_seq

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
