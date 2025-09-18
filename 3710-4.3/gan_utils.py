import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from gan_config import config

def save_samples(generator, epoch, batch_i, fixed_noise):
    """保存生成的样本图像"""
    generator.eval()
    
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    
    # 保存图像网格
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = fake_images[i].squeeze().numpy()
        img = (img + 1) / 2  # 反归一化到 [0, 1]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "samples", f"epoch_{epoch}_batch_{batch_i}.png"))
    plt.close()
    
    generator.train()

def plot_losses(g_losses, d_losses):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.savefig(os.path.join(config.output_dir, 'loss_curve.png'))
    plt.close()

def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    
    torch.save(checkpoint, os.path.join(config.output_dir, "checkpoints", f'checkpoint_epoch_{epoch}.pt'))

def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    return checkpoint['epoch']