import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from gan_config import config

class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # 遍历目录找到所有图像文件
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    self.image_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # 读取图像并转换为灰度
            image = Image.open(img_path).convert('L')
            
            if self.transform:
                image = self.transform(image)
            else:
                # 默认转换
                transform = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.CenterCrop(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
                ])
                image = transform(image)
                
            return image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为替代
            return torch.zeros((1, config.image_size, config.image_size))

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
    ])
    
    dataset = OASISDataset(config.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader