#!/bin/bash
# 单张4090 GPU训练配置
# 使用 sam2 conda 环境

# 激活conda环境
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate sam2

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# 启动训练
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_trop2_me_nu.yaml \
  --num-gpus 1
