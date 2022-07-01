#!/usr/bin/env bash

# This script is used to train a model.

# basic settings
root_dir=.
degradation=$1
model=$2
gpu_ids=0,1  # set to -1 to use cpu
master_port=4321

debug=0


# retain training or train from scratch
start_iter=0
if [[ ${start_iter} > 0 ]]; then
    suffix=_iter${start_iter}
else
    suffix=''
fi


exp_dir=${root_dir}/experiments_${degradation}/${model}
# check
if [ -d "$exp_dir/train" ]; then
    echo ">> Experiment dir already exists: $exp_dir/train"
    echo ">> Please delete it for retraining"
  	exit 1
fi
# make dir
mkdir -p ${exp_dir}/train


# backup codes
if [[ ${debug} > 0 ]]; then
    cp -r ${root_dir}/codes ${exp_dir}/train/codes_backup${suffix}
fi


# run
num_gpus=`echo ${gpu_ids} | awk -F\, '{print NF}'`
if [[ ${num_gpus} > 1 ]]; then
    dist_args="-m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port ${master_port}"
fi

CUDA_VISIBLE_DEVICES=${gpu_ids} \
  python ${dist_args} ${root_dir}/run.py \
  --exp_dir ${exp_dir} \
  --mode train \
  --opt train${suffix}.yml \
  --gpu_ids ${gpu_ids} \
  > ${exp_dir}/train/train${suffix}.log  2>&1 &

