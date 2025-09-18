#!/bin/bash
#SBATCH --job-name=vae_oasis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err

# 如果你的环境用 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch   # 或相应环境

# 调整以下路径
DATA_ROOT=/home/groups/comp3710/preprocessed_npy_slices  # 你放 slice 的目录
OUT_DIR=/home/$USER/vae_oasis_outputs

mkdir -p $OUT_DIR

python vae_oasis.py \
  --data-dir $DATA_ROOT \
  --mode npy \
  --out-dir $OUT_DIR \
  --epochs 5 \
  --batch-size 64 \
  --latent-dim 2 \
  --resize 128 \
  --mode-run train \
  --lr 1e-3
