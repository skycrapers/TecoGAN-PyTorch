#!/usr/bin/env bash

# This script is used to evaluate a pretrained model

# basic settings
root_dir=.
degradation=$1
model=$2
exp_id=001
gpu_ids=0,1
master_port=4321


# run
num_gpus=`echo ${gpu_ids} | awk -F\, '{print NF}'`
if [[ ${num_gpus} > 1 ]]; then
	dist_args="-m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port ${master_port}"
fi

CUDA_VISIBLE_DEVICES=${gpu_ids} \
  python ${dist_args} ${root_dir}/codes/main.py \
  --exp_dir ${root_dir}/experiments_${degradation}/${model}/${exp_id} \
  --mode test \
  --opt test.yml \
  --gpu_ids ${gpu_ids}

