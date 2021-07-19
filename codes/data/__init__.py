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
            'Unknown Dataset: {data_opt["name"]}'

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
            batch_size = data_opt['batch_size_per_gpu']
            shuffle = False
            sampler = DistributedSampler(dataset)
        else:
            batch_size = data_opt['batch_size_per_gpu']
            shuffle = True
            sampler = None

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
            num_workers=data_opt['num_worker_per_gpu'],
            pin_memory=data_opt['pin_memory'])

    # --- create loader for testing --- #
    elif phase == 'test':
        # create data loader
        loader = DataLoader(
            dataset=PairedFolderDataset(data_opt),
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_worker_per_gpu'],
            pin_memory=data_opt['pin_memory'])

    else:
        raise ValueError(f'Unrecognized phase: {phase}')

    return loader
