import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from data import MRISegDataset, get_file_pairs
from model_unet import UNet
from utils import dice_coef, one_hot
from tqdm import tqdm

def evaluate(model, loader, device, num_classes):
    model.eval()
    dices = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Eval'):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = softmax(logits)
            tgt_oh = one_hot(labels, num_classes)
            dice_per_class = dice_coef(probs, tgt_oh)
            dices.append(dice_per_class.cpu().numpy())
    mean_dice = np.mean(np.stack(dices, axis=0), axis=0)
    return mean_dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img_dir", type=str, required=True)
    parser.add_argument("--test_mask_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=2)  # 背景+前景
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取测试集文件
    test_imgs, test_masks = get_file_pairs(args.test_img_dir, args.test_mask_dir, img_exts=('*.png',))
    test_dataset = MRISegDataset(test_imgs, test_masks)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    model = UNet(n_channels=1, n_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 评估
    mean_dice = evaluate(model, test_loader, device, args.num_classes)
    print("Mean Dice per class:", mean_dice)
    print("Average Dice:", mean_dice.mean())
