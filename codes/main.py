import os
import os.path as osp
import math
import time

import torch

from data import create_dataloader, prepare_data
from models import define_model
from models.networks import define_generator
from metrics.metric_calculator import MetricCalculator
from metrics.model_summary import register, profile_model
from utils import dist_utils, base_utils, data_utils


def train(opt):
    # logging
    base_utils.print_options(opt, logger)

    # create data loader
    train_loader = create_dataloader(opt, phase='train', idx='train')

    # create downsampling kernels for BD degradation
    kernel = data_utils.create_kernel(opt)

    # create model
    model = define_model(opt)

    # training configs
    total_sample = len(train_loader.dataset)
    iter_per_epoch = len(train_loader)
    total_iter = opt['train']['total_iter']
    total_epoch = int(math.ceil(total_iter / iter_per_epoch))
    start_iter, iter = opt['train']['start_iter'], 0

    test_freq = opt['test']['test_freq']
    log_freq = opt['logger']['log_freq']
    ckpt_freq = opt['logger']['ckpt_freq']

    base_utils.log_info(f'Number of training samples: {total_sample}')
    base_utils.log_info(f'Total epochs needed: {total_epoch} for {total_iter} iterations')

    # train
    for epoch in range(total_epoch):
        if opt['dist']:
            train_loader.sampler.set_epoch(epoch)

        for data in train_loader:
            # update iter
            iter += 1
            curr_iter = start_iter + iter
            if iter > total_iter:
                base_utils.log_info('Finish training')
                break

            # update learning rate
            model.update_learning_rate()

            # prepare data
            data = prepare_data(opt, data, kernel)

            # train for a mini-batch
            model.train(data)

            # update running log
            model.update_running_log()

            # log
            if log_freq > 0 and iter % log_freq == 0:
                # basic info
                msg = '[epoch: {} | iter: {}'.format(epoch, curr_iter)
                for lr_type, lr in model.get_current_learning_rate().items():
                    msg += ' | {}: {:.2e}'.format(lr_type, lr)
                msg += '] '

                # loss info
                log_dict = model.get_running_log()
                msg += ', '.join([
                    '{}: {:.3e}'.format(k, v) for k, v in log_dict.items()])

                base_utils.log_info(msg)

            # save model
            if ckpt_freq > 0 and iter % ckpt_freq == 0:
                model.save(curr_iter)

            # evaluate performance
            if test_freq > 0 and iter % test_freq == 0:
                # setup model index
                model_idx = 'G_iter{}'.format(curr_iter)

                # for each testset
                for dataset_idx in sorted(opt['dataset'].keys()):
                    # use dataset with prefix `test`
                    if not dataset_idx.startswith('test'):
                        continue

                    ds_name = opt['dataset'][dataset_idx]['name']
                    base_utils.log_info(f'Testing on {dataset_idx}: {ds_name}')

                    # create data loader
                    test_loader = create_dataloader(opt, phase='test', idx=dataset_idx)

                    # define metric calculator
                    metric_calculator = MetricCalculator(opt)

                    # infer and compute metrics for each sequence
                    for data in test_loader:
                        # fetch data
                        lr_data = data['lr'][0]
                        seq_idx = data['seq_idx'][0]
                        frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]

                        # infer
                        hr_seq = model.infer(lr_data)  # thwc|rgb|uint8

                        # save results (optional)
                        if opt['test']['save_res']:
                            res_dir = osp.join(
                                opt['test']['res_dir'], ds_name, model_idx)
                            res_seq_dir = osp.join(res_dir, seq_idx)
                            data_utils.save_sequence(
                                res_seq_dir, hr_seq, frm_idx, to_bgr=True)

                        # compute metrics for the current sequence
                        true_seq_dir = osp.join(
                            opt['dataset'][dataset_idx]['gt_seq_dir'], seq_idx)
                        metric_calculator.compute_sequence_metrics(
                            seq_idx, true_seq_dir, '', pred_seq=hr_seq)

                    # save/print metrics
                    if opt['test'].get('save_json'):
                        # save results to json file
                        json_path = osp.join(
                            opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
                        metric_calculator.save_results(
                            model_idx, json_path, override=True)
                    else:
                        # print directly
                        metric_calculator.display_results()


def test(opt):
    # logging
    base_utils.print_options(opt)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # set model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]
        
        # log
        base_utils.log_info(f'{"="*40}\nTesting model: {model_idx}\n{"="*40}')

        # create model
        opt['model']['generator']['load_path'] = load_path
        model = define_model(opt)

        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # select testing dataset
            if not 'test' in dataset_idx:
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            base_utils.log_info(f'Testing on {ds_name} dataset')

            # create data loader
            test_loader = create_dataloader(opt, phase='test', idx=dataset_idx)
            test_dataset = test_loader.dataset

            # infer and store results for each sequence
            rank, world_size = dist_utils.get_dist_info()
            for idx in range(rank, len(test_dataset), world_size):
                # fetch data
                data = test_dataset[idx]

                # infer
                hr_seq = model.infer(data['lr'])  # thwc|rgb|uint8

                # save results (optional)
                if opt['test']['save_res']:
                    res_dir = osp.join(
                        opt['test']['res_dir'], ds_name, model_idx)
                    res_seq_dir = osp.join(res_dir, data['seq_idx'])
                    data_utils.save_sequence(
                        res_seq_dir, hr_seq, data['frm_idx'], to_bgr=True)

            base_utils.log_info('-' * 40)

    # logging
    base_utils.log_info('Finish testing')
    base_utils.log_info('=' * 40)


def profile(opt, lr_size, test_speed=False):
    # logging
    base_utils.print_options(opt['model']['generator'])

    # basic configs
    scale = opt['scale']
    device = torch.device(opt['device'])

    # create model
    net_G = define_generator(opt).to(device)

    # get dummy input
    lr_size = tuple(map(int, lr_size.split('x')))
    dummy_input_dict = net_G.generate_dummy_input(lr_size)
    for key in dummy_input_dict.keys():
        dummy_input_dict[key] = dummy_input_dict[key].to(device)

    # profile
    register(net_G, dummy_input_dict)
    gflops, params = profile_model(net_G)

    base_utils.log_info('-' * 40)
    base_utils.log_info('Super-resolute data from {}x{}x{} to {}x{}x{}'.format(
        *lr_size, lr_size[0], lr_size[1]*scale, lr_size[2]*scale))
    base_utils.log_info('Parameters (x10^6): {:.3f}'.format(params/1e6))
    base_utils.log_info('FLOPs (x10^9): {:.3f}'.format(gflops))
    base_utils.log_info('-' * 40)

    # test running speed
    if test_speed:
        n_test = 3
        tot_time = 0

        for i in range(n_test):
            start_time = time.time()
            with torch.no_grad():
                _ = net_G.step(**dummy_input_dict)
            end_time = time.time()
            tot_time += end_time - start_time

        base_utils.log_info('Speed (FPS): {:.3f} (averaged for {} runs)'.format(
            n_test / tot_time, n_test))
        base_utils.log_info('-' * 40)


if __name__ == '__main__':
    # --- parse arguments --- #
    args = base_utils.parse_agrs()

    # --- generic settings --- #
    # parse configs, set device, set ramdom seed
    opt = base_utils.parse_configs(args)

    # set logger
    base_utils.setup_logger('base')

    # set dirs
    base_utils.setup_paths(opt, args.mode)

    # --- train --- #
    if args.mode == 'train':
        train(opt)

    # --- test --- #
    elif args.mode == 'test':
        test(opt)

    # --- profile --- #
    elif args.mode == 'profile':
        profile(opt, args.lr_size, args.test_speed)

    else:
        raise ValueError(
            'Unrecognized mode: {} (train|test|profile)'.format(args.mode))
