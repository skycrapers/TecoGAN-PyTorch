import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .paired_lmdb_dataset import PairedLMDBDataset
from .unpaired_lmdb_dataset import UnpairedLMDBDataset
from .paired_folder_dataset import PairedFolderDataset


def create_dataloader(opt, phase, idx):
    # setup params
    data_opt = opt['dataset'].get(idx)
    degradation_type = opt['dataset']['degradation']['type']

    # --- create loader for training --- #
    if phase == 'train':
        # check dataset
        assert data_opt['name'] in ('VimeoTecoGAN', 'VimeoTecoGAN-sub'), \
            'Unknown Dataset: {}'.format(data_opt['name'])

        if degradation_type == 'BI':
            # create dataset
            dataset = PairedLMDBDataset(
                data_opt,
                scale=opt['scale'],
                tempo_extent=opt['train']['tempo_extent'],
                moving_first_frame=opt['train'].get('moving_first_frame', False),
                moving_factor=opt['train'].get('moving_factor', 1.0))

        elif degradation_type == 'BD':
            # enlarge crop size to incorporate border size
            sigma = opt['dataset']['degradation']['sigma']
            enlarged_crop_size = data_opt['crop_size'] + 2 * int(sigma * 3.0)

            # create dataset
            dataset = UnpairedLMDBDataset(
                data_opt,
                crop_size=enlarged_crop_size,  # override
                tempo_extent=opt['train']['tempo_extent'],
                moving_first_frame=opt['train'].get('moving_first_frame', False),
                moving_factor=opt['train'].get('moving_factor', 1.0))

        else:
            raise ValueError(f'Unrecognized degradation type: {degradation_type}')

        # create data loader
        if opt['dist']:
            batch_size = data_opt['batch_size'] // opt['world_size']
            shuffle = False
            sampler = DistributedSampler(dataset)
        else:
            batch_size = data_opt['batch_size']
            shuffle = True
            sampler = None

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    # --- create loader for testing --- #
    elif phase == 'test':
        # create data loader
        loader = DataLoader(
            dataset=PairedFolderDataset(data_opt),
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    else:
        raise ValueError('Unrecognized phase: {}'.format(phase))

    return loader


def prepare_data(opt, data, kernel):
    """ prepare gt, lr data for training

        for BD degradation, generate lr data and remove border of gt data
        for BI degradation, return data directly

    """

    device = torch.device(opt['device'])
    degradation_type = opt['dataset']['degradation']['type']

    if degradation_type == 'BI':
        gt_data, lr_data = data['gt'].to(device), data['lr'].to(device)

    elif degradation_type == 'BD':
        # setup params
        scale = opt['scale']
        sigma = opt['dataset']['degradation'].get('sigma', 1.5)
        border_size = int(sigma * 3.0)

        gt_with_border = data['gt'].to(device)
        n, t, c, gt_h, gt_w = gt_with_border.size()
        lr_h = (gt_h - 2 * border_size) // scale
        lr_w = (gt_w - 2 * border_size) // scale

        # generate lr data
        gt_with_border = gt_with_border.view(n * t, c, gt_h, gt_w)
        lr_data = F.conv2d(
            gt_with_border, kernel, stride=scale, bias=None, padding=0)
        lr_data = lr_data.view(n, t, c, lr_h, lr_w)

        # remove gt border
        gt_data = gt_with_border[
            ...,
            border_size: border_size + scale * lr_h,
            border_size: border_size + scale * lr_w
        ]
        gt_data = gt_data.view(n, t, c, scale * lr_h, scale * lr_w)

    else:
        raise ValueError(f'Unrecognized degradation type: {degradation_type}')

    return { 'gt': gt_data, 'lr': lr_data }
