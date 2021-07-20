from collections import OrderedDict

import torch
import torch.optim as optim
import torch.distributed as dist

from .vsr_model import VSRModel
from .networks import define_generator, define_discriminator
from .networks.vgg_nets import VGGFeatureExtractor
from .optim import define_criterion, define_lr_schedule
from utils import base_utils, net_utils, dist_utils


class VSRGANModel(VSRModel):
    """ A model wrapper for subjective video super-resolution
    """

    def __init__(self, opt):
        super(VSRGANModel, self).__init__(opt)

        if self.is_train:
            self.cnt_upd_D = 0

    def set_networks(self):
        # define generator
        self.net_G = define_generator(self.opt)
        self.net_G = self.model_to_device(self.net_G)
        base_utils.log_info('Generator: {}\n{}'.format(
            self.opt['model']['generator']['name'], self.net_G.__str__()))

        # load generator
        load_path_G = self.opt['model']['generator'].get('load_path', '')
        if load_path_G:
            self.load_network(self.net_G, load_path_G)
            base_utils.log_info('Load generator from: {}'.format(load_path_G))

        if self.is_train:
            # define discriminator
            self.net_D = define_discriminator(self.opt)
            self.net_D = self.model_to_device(self.net_D)
            base_utils.log_info('Discriminator: {}\n{}'.format(
                self.opt['model']['discriminator']['name'], self.net_D.__str__()))

            # load discriminator
            load_path_D = self.opt['model']['discriminator'].get('load_path', '')
            if load_path_D:
                self.load_network(self.net_D, load_path_D)
                base_utils.log_info('Load discriminator from: {}'.format(load_path_D))

    def set_criterions(self):
        # pixel criterion
        self.pix_crit = define_criterion(self.opt['train'].get('pixel_crit'))

        # warping criterion
        self.warp_crit = define_criterion(self.opt['train'].get('warping_crit'))

        # feature criterion
        self.feat_crit = define_criterion(
            self.opt['train'].get('feature_crit'))
        if self.feat_crit is not None:  # load feature extractor
            feature_layers = self.opt['train']['feature_crit'].get(
                'feature_layers', [8, 17, 26, 35])
            self.net_F = VGGFeatureExtractor(feature_layers).to(self.device)

        # ping-pong criterion
        self.pp_crit = define_criterion(self.opt['train'].get('pingpong_crit'))

        # feature matching criterion
        self.fm_crit = define_criterion(self.opt['train'].get('feature_matching_crit'))

        # gan criterion
        self.gan_crit = define_criterion(self.opt['train'].get('gan_crit'))

    def set_optimizers(self):
        # set optimizer for net_G
        self.optim_G = optim.Adam(
            self.net_G.parameters(),
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=self.opt['train']['generator'].get('betas', (0.9, 0.999)))

        # set optimizer for net_D
        self.optim_D = optim.Adam(
            self.net_D.parameters(),
            lr=self.opt['train']['discriminator']['lr'],
            weight_decay=self.opt['train']['discriminator'].get('weight_decay', 0),
            betas=self.opt['train']['discriminator'].get('betas', (0.9, 0.999)))

    def set_lr_schedules(self):
        # set lr schedules for net_G
        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

        # set lr schedules for net_D
        self.sched_D = define_lr_schedule(
            self.opt['train']['discriminator'].get('lr_schedule'), self.optim_D)

    def train(self):
        # === prepare data === #
        lr_data, gt_data = self.lr_data, self.gt_data

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        # generate bicubic upsampled data
        upsample_fn = self.get_bare_model(self.net_G).upsample_func
        bi_data = upsample_fn(
            lr_data.view(n * t, c, lr_h, lr_w)).view(n, t, c, gt_h, gt_w)

        # augment data for pingpong criterion
        # i.e., (0,1,2,...,t-2,t-1) -> (0,1,2,...,t-2,t-1,t-2,...,2,1,0)
        if self.pp_crit is not None:
            lr_rev = lr_data.flip(1)[:, 1:, ...]
            gt_rev = gt_data.flip(1)[:, 1:, ...]
            bi_rev = bi_data.flip(1)[:, 1:, ...]

            lr_data = torch.cat([lr_data, lr_rev], dim=1)
            gt_data = torch.cat([gt_data, gt_rev], dim=1)
            bi_data = torch.cat([bi_data, bi_rev], dim=1)

        # === initialize === #
        self.net_G.train()
        self.net_D.train()
        self.optim_G.zero_grad()
        self.optim_D.zero_grad()
        self.log_dict = OrderedDict()

        # === forward net_G === #
        net_G_output_dict = self.net_G(lr_data)
        hr_data = net_G_output_dict['hr_data']

        # === forward net_D === #
        for param in self.net_D.parameters():
            param.requires_grad = True

        # feed additional data
        net_D_input_dict = {
            'net_G': self.get_bare_model(self.net_G),  # TODO: check
            'lr_data': lr_data,
            'bi_data': bi_data,
            'use_pp_crit': (self.pp_crit is not None),
            'crop_border_ratio': self.opt['train']['discriminator'].get(
                'crop_border_ratio', 1.0)
        }
        net_D_input_dict.update(net_G_output_dict)

        # forward real sequence (gt)
        real_pred, net_D_oputput_dict = self.net_D(gt_data, net_D_input_dict)

        # reuse internal data (e.g., optical flow)
        net_D_input_dict.update(net_D_oputput_dict)

        # forward fake sequence (hr)
        fake_pred, _ = self.net_D(hr_data.detach(), net_D_input_dict)

        # === optimize net_D === #
        real_pred_D, fake_pred_D = real_pred[0], fake_pred[0]

        # select D update policy
        update_policy = self.opt['train']['discriminator']['update_policy']
        if update_policy == 'adaptive':
            # update D adaptively
            logged_real_pred_D = torch.log(torch.sigmoid(real_pred_D) + 1e-8).mean()
            logged_fake_pred_D = torch.log(torch.sigmoid(fake_pred_D) + 1e-8).mean()

            if self.dist:
                # synchronize
                dist.all_reduce(logged_real_pred_D)
                dist.all_reduce(logged_fake_pred_D)
                dist.barrier()

                logged_real_pred_D /= self.opt['world_size']
                logged_fake_pred_D /= self.opt['world_size']

            distance = (logged_real_pred_D - logged_fake_pred_D).item()
            upd_D = distance < self.opt['train']['discriminator']['update_threshold']
        else:
            upd_D = True

        if upd_D:
            self.cnt_upd_D += 1.0
            real_loss_D = self.gan_crit(real_pred_D, True)
            fake_loss_D = self.gan_crit(fake_pred_D, False)
            loss_D = real_loss_D + fake_loss_D

            # update net_D
            loss_D.backward()
            self.optim_D.step()
        else:
            loss_D = torch.zeros(1)

        # logging
        self.log_dict['l_gan_D'] = loss_D.item()
        self.log_dict['p_real_D'] = real_pred_D.mean().item()
        self.log_dict['p_fake_D'] = fake_pred_D.mean().item()
        if update_policy == 'adaptive':
            self.log_dict['distance'] = distance
            self.log_dict['n_upd_D'] = self.cnt_upd_D

        # === optimize net_G === #
        for param in self.net_D.parameters():
            param.requires_grad = False

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
            loss_pix_G = pix_w * self.pix_crit(hr_data, gt_data)
            loss_G += loss_pix_G
            self.log_dict['l_pix_G'] = loss_pix_G.item()

        # warping (warp) loss
        if self.warp_crit is not None:
            lr_curr = net_G_output_dict['lr_curr']
            lr_prev = net_G_output_dict['lr_prev']
            lr_flow = net_G_output_dict['lr_flow']
            lr_warp = net_utils.backward_warp(lr_prev, lr_flow)

            warp_w = self.opt['train']['warping_crit'].get('weight', 1)
            loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr)
            loss_G += loss_warp_G
            self.log_dict['l_warp_G'] = loss_warp_G.item()

        # feature/perceptual (feat) loss
        if self.feat_crit is not None:
            hr_merge = hr_data.view(-1, c, gt_h, gt_w)
            gt_merge = gt_data.view(-1, c, gt_h, gt_w)

            hr_feat_lst = self.net_F(hr_merge)
            gt_feat_lst = self.net_F(gt_merge)

            loss_feat_G = 0
            for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
                loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())

            feat_w = self.opt['train']['feature_crit'].get('weight', 1)
            loss_feat_G = feat_w * loss_feat_G
            loss_G += loss_feat_G
            self.log_dict['l_feat_G'] = loss_feat_G.item()

        # ping-pong (pp) loss
        if self.pp_crit is not None:
            tempo_extent = self.opt['train']['tempo_extent']
            hr_data_fw = hr_data[:, :tempo_extent - 1, ...]      # -------->|
            hr_data_bw = hr_data[:, tempo_extent:, ...].flip(1)  # <--------|

            pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
            loss_pp_G = pp_w * self.pp_crit(hr_data_fw, hr_data_bw)
            loss_G += loss_pp_G
            self.log_dict['l_pp_G'] = loss_pp_G.item()

        # feature matching (fm) loss
        if self.fm_crit is not None:
            fake_pred, _ = self.net_D(hr_data, net_D_input_dict)
            fake_feat_lst, real_feat_lst = fake_pred[-1], real_pred[-1]

            layer_norm = self.opt['train']['feature_matching_crit'].get(
                'layer_norm', [12.0, 14.0, 24.0, 100.0])

            loss_fm_G = 0
            for i in range(len(real_feat_lst)):
                fake_feat, real_feat = fake_feat_lst[i], real_feat_lst[i]
                loss_fm_G += self.fm_crit(
                    fake_feat, real_feat.detach()) / layer_norm[i]

            fm_w = self.opt['train']['feature_matching_crit'].get('weight', 1)
            loss_fm_G = fm_w * loss_fm_G
            loss_G += loss_fm_G
            self.log_dict['l_fm_G'] = loss_fm_G.item()

        # gan loss
        if self.fm_crit is None:
            fake_pred, _ = self.net_D(hr_data, net_D_input_dict)
        fake_pred_G = fake_pred[0]

        gan_w = self.opt['train']['gan_crit'].get('weight', 1)
        loss_gan_G = gan_w * self.gan_crit(fake_pred_G, True)
        loss_G += loss_gan_G
        self.log_dict['l_gan_G'] = loss_gan_G.item()
        self.log_dict['p_fake_G'] = fake_pred_G.mean().item()

        # update net_G
        loss_G.backward()
        self.optim_G.step()

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
        self.save_network(self.net_D, 'D', current_iter)
