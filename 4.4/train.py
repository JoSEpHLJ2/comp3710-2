import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from data import MRISegDataset, get_file_pairs
from model_unet import UNet
from utils import dice_coef, one_hot, save_overlay
from tqdm import tqdm
import argparse

# -------------------------
# 训练函数
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc='Train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

# -------------------------
# 验证函数
# -------------------------
def validate(model, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    dices = []
    softmax = torch.nn.Softmax(dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val'):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            probs = softmax(logits)
            tgt_oh = one_hot(labels, num_classes)
            dice_per_class = dice_coef(probs, tgt_oh)
            dices.append(dice_per_class.cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    mean_dice = np.mean(np.stack(dices, axis=0), axis=0)

    # 打印前几个类别的 Dice
    num_show = min(num_classes, 5)
    dice_str = ", ".join([f"{d:.4f}" for d in mean_dice[:num_show]])
    print(f"Val Dice (first {num_show} classes): {dice_str} ... | Mean: {mean_dice.mean():.4f}")

    return mean_loss, mean_dice

# -------------------------
# 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', required=True)
    parser.add_argument('--train_mask_dir', required=True)
    parser.add_argument('--val_img_dir', required=True)
    parser.add_argument('--val_mask_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 数据集
    train_imgs, train_labels = get_file_pairs(args.train_img_dir, args.train_mask_dir)
    val_imgs, val_labels = get_file_pairs(args.val_img_dir, args.val_mask_dir)
    train_ds = MRISegDataset(train_imgs, train_labels)
    val_ds = MRISegDataset(val_imgs, val_labels)

    # 自动检测类别数
    max_label = 0
    for _, lbl in train_ds:
        max_label = max(max_label, int(lbl.max()))
    for _, lbl in val_ds:
        max_label = max(max_label, int(lbl.max()))
    num_classes = max_label + 1
    print(f'Detected {num_classes} classes.')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 模型和设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")   # 打印设备
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    model = UNet(in_ch=1, out_ch=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(args.epochs):
        print(f'\n=== Epoch {epoch+1}/{args.epochs} ===')
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, device, num_classes=num_classes)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'unet_best.pt'))

        # 保存每个 epoch 的模型
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'unet_epoch{epoch+1}.pt'))

        # 可视化前几个验证样本
        imgs, labels = next(iter(val_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(imgs)
            probs = torch.nn.Softmax(dim=1)(logits)
            for i in range(min(2, imgs.size(0))):
                save_overlay(imgs[i], probs[i], labels[i], os.path.join(args.out_dir, f'overlay_epoch{epoch+1}_{i}.png'))

if __name__ == '__main__':
    main()
