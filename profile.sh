#!/usr/bin/env bash

# This script is used to profile a model.

# basic settings
root_dir=.
degradation=$1
model=$2
gpu_ids=0
# specify the size of input data in the format of [color]x[height]x[weight]
lr_size=$3


# run
python ${root_dir}/run.py \
  --exp_dir ${root_dir}/experiments_${degradation}/${model} \
  --mode profile \
  --opt test.yml \
  --gpu_ids ${gpu_ids} \
  --lr_size ${lr_size} \
  --test_speed

