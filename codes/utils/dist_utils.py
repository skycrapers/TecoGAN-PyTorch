import functools

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(opt, local_rank):
    """ Adopted from BasicSR
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    rank, world_size = get_dist_info()

    opt.update({
        'dist': True,
        'device': 'cuda',
        'local_rank': local_rank,
        'world_size': world_size,
        'rank': rank
    })


def get_dist_info():
    """ Adopted from BasicSR
    """
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def master_only(func):
    """ Adopted from BasicSR
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
