#!/usr/bin/env bash

# This script is used to evaluate a pretrained model.

# basic settings
root_dir="`realpath .`"
if [ ! -d "$root_dir/codes" ]; then
    echo "This script must run from the root of the TecoGAN-PyTorch repo directory containing the codes module."
    exit 1
fi
degradation=$1
model=$2
gpu_ids=2,3
master_port=4322


# run
num_gpus=`echo ${gpu_ids} | awk -F\, '{print NF}'`
if [[ ${num_gpus} > 1 ]]; then
	dist_args="-m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port ${master_port}"
fi

CUDA_VISIBLE_DEVICES=${gpu_ids} \
  python ${dist_args} ${root_dir}/run.py \
  --exp_dir ${root_dir}/experiments_${degradation}/${model} \
  --mode test \
  --opt test.yml \
  --gpu_ids ${gpu_ids}

