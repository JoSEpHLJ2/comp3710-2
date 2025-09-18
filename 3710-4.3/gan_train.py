import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from gan_model import Generator, Discriminator, weights_init
from gan_data import get_dataloader
from gan_config import config

def save_samples_short(generator, epoch, batch_i, fixed_noise):
    """快速保存样本"""
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    
    # 只保存一张样本图
    img = fake_images[0].squeeze().numpy()
    img = (img + 1) / 2  # 反归一化到 [0, 1]
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Epoch {epoch}, Batch {batch_i}')
    plt.savefig(os.path.join(config.output_dir, "samples", f"sample_epoch{epoch}_batch{batch_i}.png"))
    plt.close()
    
    generator.train()

def train_gan_short():
    print("🚀 开始短期GAN训练 (3轮)")
    
    # 初始化模型
    generator = Generator().to(config.device)
    discriminator = Discriminator().to(config.device)
    
    # 初始化权重
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # 使用更稳定的设置
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # 数据加载器
    dataloader = get_dataloader()
    
    # 固定噪声用于生成样本
    fixed_noise = torch.randn(16, config.latent_dim, 1, 1, device=config.device)
    
    print(f"设备: {config.device}")
    print(f"每轮批次数量: {len(dataloader)}")
    print(f"总轮数: {config.epochs}")
    
    for epoch in range(config.epochs):
        start_time = time.time()
        epoch_d_losses = []
        epoch_g_losses = []
        
        print(f"\n=== 第 {epoch+1}/{config.epochs} 轮 ===")
        
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(config.device)
            
            # 使用软标签
            real_labels = torch.full((batch_size,), 0.9, device=config.device)
            fake_labels = torch.full((batch_size,), 0.1, device=config.device)
            
            # ---------------------
            # 训练判别器
            # ---------------------
            discriminator.zero_grad()
            
            # 真实图像
            outputs_real = discriminator(real_images)
            d_loss_real = criterion(outputs_real, real_labels)
            
            # 假图像
            noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=config.device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            # 训练生成器
            # ---------------------
            generator.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # 记录损失
            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(g_loss.item())
            
            # 每50个batch打印一次
            if i % 50 == 0:
                print(f'批次 [{i}/{len(dataloader)}] D损失: {d_loss.item():.4f} G损失: {g_loss.item():.4f}')
            
            # 每100个batch保存样本
            if i % 100 == 0:
                save_samples_short(generator, epoch+1, i, fixed_noise)
        
        # 每轮结束统计
        epoch_time = time.time() - start_time
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_loss = np.mean(epoch_g_losses)
        
        print(f"第 {epoch+1} 轮完成, 耗时: {epoch_time:.2f}s")
        print(f"平均D损失: {avg_d_loss:.4f}, 平均G损失: {avg_g_loss:.4f}")
        
        # 保存检查点
        torch.save(generator.state_dict(), os.path.join(config.output_dir, "checkpoints", f'generator_epoch{epoch+1}.pt'))
        torch.save(discriminator.state_dict(), os.path.join(config.output_dir, "checkpoints", f'discriminator_epoch{epoch+1}.pt'))
    
    # 保存最终模型
    torch.save(generator.state_dict(), os.path.join(config.output_dir, 'generator_final.pt'))
    torch.save(discriminator.state_dict(), os.path.join(config.output_dir, 'discriminator_final.pt'))
    
    print("\n✅ 3轮训练完成！")
    print(f"结果保存在: {config.output_dir}")

if __name__ == "__main__":
    train_gan_short()