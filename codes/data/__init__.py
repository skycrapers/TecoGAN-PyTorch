import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .paired_lmdb_dataset import PairedLMDBDataset
from .unpaired_lmdb_dataset import UnpairedLMDBDataset
from .paired_folder_dataset import PairedFolderDataset
from .unpaired_folder_dataset import UnpairedFolderDataset


def create_dataloader(opt, phase, idx):
    # set params
    data_opt = opt['dataset'].get(idx)
    degradation_type = opt['dataset']['degradation']['type']

    # === create loader for training === #
    if phase == 'train':
        # check dataset
        assert data_opt['name'] in ('VimeoTecoGAN', 'REDS'), \
            f'Unknown Dataset: {data_opt["name"]}'

        if degradation_type == 'BI':
            # create dataset
            dataset = PairedLMDBDataset(
                data_opt,
                scale=opt['scale'],
                tempo_extent=opt['train']['tempo_extent'],
                moving_first_frame=opt['train'].get('moving_first_frame', False),
                moving_factor=opt['train'].get('moving_factor', 1.0))

        elif degradation_type == 'BD':
            # enlarge the crop size to incorporate border
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

    # === create loader for testing === #
    elif phase == 'test':
        # create data loader
        if 'lr_seq_dir' in data_opt and data_opt['lr_seq_dir']:
            loader = DataLoader(
                dataset=PairedFolderDataset(data_opt),
                batch_size=1,
                shuffle=False,
                num_workers=data_opt['num_worker_per_gpu'],
                pin_memory=data_opt['pin_memory'])

        else:
            assert degradation_type == 'BD', \
                '"lr_seq_dir" is required for BI mode'

            sigma = opt['dataset']['degradation']['sigma']
            ksize = 2 * int(sigma * 3.0) + 1

            loader = DataLoader(
                dataset=UnpairedFolderDataset(
                    data_opt, scale=opt['scale'], sigma=sigma, ksize=ksize),
                batch_size=1,
                shuffle=False,
                num_workers=data_opt['num_worker_per_gpu'],
                pin_memory=data_opt['pin_memory'])

    else:
        raise ValueError(f'Unrecognized phase: {phase}')

    return loader
