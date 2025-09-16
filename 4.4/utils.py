import torch
import numpy as np
import matplotlib.pyplot as plt

def one_hot(labels, num_classes):
    B,H,W = labels.shape
    oh = torch.zeros((B,num_classes,H,W), device=labels.device)
    oh.scatter_(1, labels.unsqueeze(1), 1)
    return oh

def dice_coef(pred_probs, target_onehot, eps=1e-7):
    intersection = (pred_probs * target_onehot).sum(dim=(2,3))
    union = pred_probs.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean(dim=0)

def save_overlay(img_tensor, pred_probs, target, out_path):
    pred_classes = pred_probs.argmax(0).cpu().numpy()
    img = img_tensor.squeeze(0).cpu().numpy()
    H,W = img.shape
    rgb = np.stack([img,img,img], axis=-1)
    cmap = np.array([
        [1,0,0],   # red
        [0,1,0],   # green
        [0,0,1],   # blue
        [1,1,0],   # yellow
        [1,0,1]    # magenta
    ])
    overlay = rgb.copy()
    for c in range(min(pred_probs.shape[0], cmap.shape[0])):
        mask = pred_classes == c
        overlay[mask] = overlay[mask]*0.5 + cmap[c]*0.5
    plt.imsave(out_path, overlay)
