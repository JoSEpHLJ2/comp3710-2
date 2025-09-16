#!/bin/bash
#SBATCH --job-name=cifar10
#SBATCH --output=cifar10.out
#SBATCH --error=cifar10.err
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00

# 激活 Python 虚拟环境
source ~/venv3710/bin/activate

# 运行训练脚本
python /home/Student/s4908583/cifar10_dawnbench/train_cifar10.py \
    --epochs 100 \
    --batch-size 512 \
    --lr 0.1
