import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# -------------------------
# 匹配图像和mask
# -------------------------
def get_file_pairs(img_dir, mask_dir):
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
    assert len(img_files) == len(mask_files), "图像和mask数量不一致"
    return img_files, mask_files

# -------------------------
# 数据集类
# -------------------------
class MRISegDataset(Dataset):
    def __init__(self, img_files, mask_files, transform=None):
        self.img_files = img_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 读取图像 (灰度，范围 [0,1])
        img = Image.open(self.img_files[idx]).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # (1,H,W)

        # 读取mask (灰度 -> 类别索引)
        mask = Image.open(self.mask_files[idx]).convert('L')
        mask = np.array(mask, dtype=np.int64)

        # 判断类别数量
        unique_vals = np.unique(mask)
        if set(unique_vals.tolist()).issubset({0, 255}):
            # 二分类 (0=背景, 255=前景)
            mask = (mask > 127).astype(np.int64)
        else:
            # 多分类 -> 映射成 0,1,2,...
            val_map = {v: i for i, v in enumerate(unique_vals)}
            mask = np.vectorize(val_map.get)(mask)

        mask = torch.tensor(mask, dtype=torch.long)  # (H,W)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask
