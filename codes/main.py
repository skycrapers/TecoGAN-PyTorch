import os
import os.path as osp
import math
import time

import torch

from data import create_dataloader
from models import define_model
from models.networks import define_generator
from metrics.metric_calculator import MetricCalculator
from utils import dist_utils, base_utils, data_utils


def train(opt):
    # print configurations
    base_utils.log_info(f'{20*"-"} Configurations {20*"-"}')
    base_utils.print_options(opt)

    # create data loader
    train_loader = create_dataloader(opt, phase='train', idx='train')

    # build model
    model = define_model(opt)

    # set training params
    total_sample, iter_per_epoch = len(train_loader.dataset), len(train_loader)
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
            if iter > total_iter: break

            # update learning rate
            model.update_learning_rate()

            # prepare data
            model.prepare_data(data)

            # train a mini-batch
            model.train()

            # update running log
            model.reduce_log()  # for distributed training
            model.update_running_log()

            # print messages
            if log_freq > 0 and curr_iter % log_freq == 0:
                msg = model.get_format_msg(epoch, curr_iter)
                base_utils.log_info(msg)

            # save model
            if ckpt_freq > 0 and curr_iter % ckpt_freq == 0:
                model.save(curr_iter)

            # evaluate model
            if test_freq > 0 and curr_iter % test_freq == 0:
                # set model index
                model_idx = f'G_iter{curr_iter}'

                # for each testset
                for dataset_idx in sorted(opt['dataset'].keys()):
                    # select test dataset
                    if 'test' not in dataset_idx: continue

                    ds_name = opt['dataset'][dataset_idx]['name']
                    base_utils.log_info(f'Testing on {ds_name} dataset')

                    # create metric calculator
                    metric_calculator = MetricCalculator(opt)

                    # create data loader
                    test_loader = create_dataloader(
                        opt, phase='test', idx=dataset_idx)
                    test_dataset = test_loader.dataset
                    num_seq = len(test_dataset)

                    # infer and compute metrics
                    rank, world_size = dist_utils.get_dist_info()
                    for i in range(rank, num_seq, world_size):
                        # fetch data
                        data = test_dataset[i]

                        # infer
                        hr_seq = model.infer(data['lr'])

                        # save hr results (optional)
                        if opt['test']['save_res']:
                            res_dir = osp.join(
                                opt['test']['res_dir'], ds_name, model_idx)
                            res_seq_dir = osp.join(res_dir, data['seq_idx'])
                            data_utils.save_sequence(
                                res_seq_dir, hr_seq, data['frm_idx'], to_bgr=True)

                        # compute metrics for the current sequence
                        true_seq_dir = osp.join(
                            opt['dataset'][dataset_idx]['gt_seq_dir'], data['seq_idx'])
                        metric_calculator.compute_sequence_metrics(
                            data['seq_idx'], true_seq_dir, '', pred_seq=hr_seq)

                    # save/print results
                    if opt['test'].get('save_json'):
                        # write to a json file
                        json_path = osp.join(
                            opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
                        metric_calculator.get_averaged_results(num_seq)
                        metric_calculator.save_results(
                            model_idx, json_path, override=True)
                    else:
                        # print directly
                        metric_calculator.get_averaged_results(num_seq)
                        metric_calculator.display_results()


def test(opt):
    # logging
    base_utils.print_options(opt)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # set model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]
        
        # log
        base_utils.log_info(f'\n{"="*40}')
        base_utils.log_info(f'Testing model: {model_idx}')
        base_utils.log_info(f'{"="*40}')

        # create model
        opt['model']['generator']['load_path'] = load_path
        model = define_model(opt)

        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # select testing dataset
            if 'test' not in dataset_idx:
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            base_utils.log_info(f'Testing on {ds_name} dataset')

            # create data loader
            test_loader = create_dataloader(opt, phase='test', idx=dataset_idx)
            test_dataset = test_loader.dataset

            # infer and save results
            rank, world_size = dist_utils.get_dist_info()
            for idx in range(rank, len(test_dataset), world_size):
                # fetch data
                data = test_dataset[idx]

                # infer
                hr_seq = model.infer(data['lr'])

                # save hr results (optional)
                if opt['test']['save_res']:
                    res_dir = osp.join(
                        opt['test']['res_dir'], ds_name, model_idx)
                    res_seq_dir = osp.join(res_dir, data['seq_idx'])
                    data_utils.save_sequence(
                        res_seq_dir, hr_seq, data['frm_idx'], to_bgr=True)

            base_utils.log_info('-'*40)

    # logging
    base_utils.log_info(f'Finish testing\n{"="*40}')


def profile(opt, lr_size, test_speed=False):
    # basic configs
    scale = opt['scale']
    device = torch.device(opt['device'])
    msg = '\n'

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False

    # logging
    base_utils.print_options(opt['model']['generator'])
    msg += f'{"*"*40}\nOriginal resolution: {lr_size} (To perform {scale}x SR)'

    # create model
    net_G = define_generator(opt).to(device)
    # base_utils.log_info(f'\n{net_G.__str__()}')

    # profile
    lr_size = tuple(map(int, lr_size.split('x')))
    gflops_dict, params_dict = net_G.profile(lr_size, device)

    gflops_all, params_all = 0, 0
    for module_name in gflops_dict.keys():
        gflops, params = gflops_dict[module_name], params_dict[module_name]
        msg += f'\n{"-"*40}\nModule: [{module_name}]'
        msg += f'\n    FLOPs (10^9): {gflops:.3f}'
        msg += f'\n    Parameters (10^6): {params/1e6:.3f}'
        gflops_all += gflops
        params_all += params
    msg += f'\n{"-"*40}\nOverall'
    msg += f'\n    FLOPs (10^9): {gflops_all:.3f}'
    msg += f'\n    Parameters (10^6): {params_all/1e6:.3f}\n{"*"*40}'

    # test running speed
    if test_speed:
        n_test, tot_time = 30, 0
        for i in range(n_test):
            dummy_input_list = net_G.generate_dummy_data(lr_size, device)

            start_time = time.time()
            # ---
            net_G.eval()
            with torch.no_grad():
                _ = net_G.step(*dummy_input_list)
            torch.cuda.synchronize()
            # ---
            end_time = time.time()
            tot_time += end_time - start_time
        msg += f'\nSpeed: {n_test/tot_time:.3f} FPS (averaged over {n_test} runs)\n{"*"*40}'

    base_utils.log_info(msg)


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
        raise ValueError(f'Unrecognized mode: {args.mode} (train|test|profile)')
