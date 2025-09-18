#!/bin/bash
#SBATCH --job-name=oasis_gan
#SBATCH --output=gan_train.out
#SBATCH --error=gan_train.err
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00

# 激活虚拟环境
source ~/venv_unet/bin/activate

# 运行GAN训练
python gan_train.py