#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --output=unet.out
#SBATCH --error=unet.err
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00

# 激活虚拟环境
source ~/venv_unet/bin/activate

# 运行训练
python3 train.py \
    --train_img_dir keras_png_slices_data/keras_png_slices_train \
    --train_mask_dir keras_png_slices_data/keras_png_slices_seg_train \
    --val_img_dir keras_png_slices_data/keras_png_slices_validate \
    --val_mask_dir keras_png_slices_data/keras_png_slices_seg_validate \
    --out_dir outputs \
    --epochs 3 \
    --batch_size 8 \
    --lr 0.0005
