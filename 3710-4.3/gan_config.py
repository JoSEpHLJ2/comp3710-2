import torch
import os

class Config:
    def __init__(self):
        # 数据配置
        self.data_dir = "keras_png_slices_data/"
        self.image_size = 64
        self.channels = 1  # 灰度图像
        
        # 模型配置
        self.latent_dim = 100
        self.gen_features = 64
        self.disc_features = 64
        
        # 训练配置
        self.batch_size = 32
        self.lr = 0.0002
        self.beta1 = 0.5
        self.epochs = 3  # 3轮训练
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 输出配置
        self.output_dir = "gan_outputs"
        self.sample_interval = 50
        self.save_interval = 1
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

# 创建配置实例
config = Config()