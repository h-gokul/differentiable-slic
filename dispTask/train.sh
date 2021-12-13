#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --gres=gpu:2

module add cuda/11.2.2
module add cudnn/v8.2.1
module add Python3/3.6.10

srun bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_superpixelnet.py"