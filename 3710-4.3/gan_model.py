import torch
import torch.nn as nn
from gan_config import config

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(config.latent_dim, config.gen_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.gen_features * 8),
            nn.ReLU(True),
            
            # 4x4
            nn.ConvTranspose2d(config.gen_features * 8, config.gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_features * 4),
            nn.ReLU(True),
            
            # 8x8
            nn.ConvTranspose2d(config.gen_features * 4, config.gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_features * 2),
            nn.ReLU(True),
            
            # 16x16
            nn.ConvTranspose2d(config.gen_features * 2, config.gen_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_features),
            nn.ReLU(True),
            
            # 32x32
            nn.ConvTranspose2d(config.gen_features, config.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: channels x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入: channels x 64 x 64
            nn.Conv2d(config.channels, config.disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32
            nn.Conv2d(config.disc_features, config.disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16
            nn.Conv2d(config.disc_features * 2, config.disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8
            nn.Conv2d(config.disc_features * 4, config.disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4
            nn.Conv2d(config.disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)