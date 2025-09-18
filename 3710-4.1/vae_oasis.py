# vae_oasis.py (完整 PNG 支持版)
import os
import glob
import argparse
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
from PIL import Image

# optional nibabel for nifti
try:
    import nibabel as nib
    HAS_NIB = True
except Exception:
    HAS_NIB = False

# Optional UMAP
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ----------------------------
# Dataset: supports .npy, .nii/.nii.gz or PNG (递归扫描子文件夹)
# ----------------------------
class OASIS2DSlices(Dataset):
    def __init__(self, root, mode='npy', transform=None, max_slices_per_subject=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.max_slices_per_subject = max_slices_per_subject

        # 递归扫描所有文件
        self.files = sorted(glob.glob(os.path.join(root, '**', '*'), recursive=True))
        self.slices = []

        if mode == 'npy':
            self.slices = [f for f in self.files if f.lower().endswith('.npy')]
        elif mode == 'png':
            self.slices = [f for f in self.files if f.lower().endswith('.png')]
        elif mode == 'nifti':
            if not HAS_NIB:
                raise RuntimeError("nibabel not installed; pip install nibabel to use nifti mode")
            self.slices = [f for f in self.files if f.endswith('.nii') or f.endswith('.nii.gz')]
        else:
            raise ValueError("mode must be 'npy', 'png', or 'nifti'")

        if len(self.slices) == 0:
            raise RuntimeError(f"No files found in {root} with mode={mode}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        path = self.slices[idx]
        if self.mode == 'npy':
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
        elif self.mode == 'png':
            img = Image.open(path).convert('L')
            arr = np.array(img, dtype=np.float32) / 255.0
        else:
            img = nib.load(path).get_fdata()
            z = img.shape[2] // 2
            if self.max_slices_per_subject and self.max_slices_per_subject > 0:
                z = random.randint(0, img.shape[2]-1)
            arr = img[:, :, z].astype(np.float32)
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = arr / (arr.max() + 1e-8)

        arr = np.expand_dims(arr, 0)  # 1,H,W
        if self.transform:
            arr = self.transform(torch.from_numpy(arr))
        else:
            arr = torch.from_numpy(arr)
        return arr

# ----------------------------
# VAE模型
# ----------------------------
class ConvEncoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ConvDecoder(nn.Module):
    def __init__(self, out_ch=1, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*4*4)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up4 = nn.ConvTranspose2d(32, out_ch, 4, 2, 1)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = self.act(self.up1(x))
        x = self.act(self.up2(x))
        x = self.act(self.up3(x))
        x = self.sig(self.up4(x))
        return x

class VAE(nn.Module):
    def __init__(self, in_ch=1, latent_dim=2):
        super().__init__()
        self.enc = ConvEncoder(in_ch, latent_dim)
        self.dec = ConvDecoder(in_ch, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        rec = self.dec(z)
        return rec, mu, logvar, z

def vae_loss(recon_x, x, mu, logvar, kld_weight=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_weight * kld, recon_loss, kld

# ----------------------------
# Training
# ----------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    transform = None
    if args.resize:
        transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((args.resize, args.resize)),
        ])

    dataset = OASIS2DSlices(args.data_dir, mode=args.mode, transform=transform)
    n = len(dataset)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    split = int(n * args.val_fraction)
    train_idx, val_idx = indices[split:], indices[:split]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    vae = VAE(in_ch=1, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    best_val = 1e9
    history = {'train_loss':[], 'val_loss':[]}

    for epoch in range(1, args.epochs+1):
        vae.train()
        running = 0.0
        for xb in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad()
            rec, mu, logvar, _ = vae(xb)
            loss, recon_l, kld = vae_loss(rec, xb, mu, logvar, kld_weight=args.kld_weight)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_loss = running / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # validation
        vae.eval()
        vrunning = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device, non_blocking=True)
                rec, mu, logvar, _ = vae(xb)
                loss, recon_l, kld = vae_loss(rec, xb, mu, logvar, kld_weight=args.kld_weight)
                vrunning += loss.item() * xb.size(0)
            val_loss = vrunning / len(val_loader.dataset)
            history['val_loss'].append(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': vae.state_dict(),
            'opt_state': optimizer.state_dict(),
            'history': history,
        }
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(args.out_dir, f"vae_epoch{epoch}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "vae_best.pt"))

    plt.figure()
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(args.out_dir, "loss.png"))
    print("Training finished. Best val:", best_val)

# ----------------------------
# Demo / reconstruction
# ----------------------------
def reconstruct_and_save(checkpoint, sample_dir, device='cpu', mode='npy', data_dir=None, resize=None):
    ck = torch.load(checkpoint, map_location=device)
    vae = VAE(in_ch=1, latent_dim=ck['model_state']['enc.fc_mu.weight'].shape[0]).to(device)
    vae.load_state_dict(ck['model_state'])
    vae.eval()
    os.makedirs(sample_dir, exist_ok=True)

    imgs = []
    if data_dir:
        if mode == 'npy':
            files = sorted(glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True))[:8]
            for f in files:
                im = np.load(f).astype(np.float32)
                im = (im - im.min())/(im.max()+1e-8)
                imgs.append(np.expand_dims(im,0))
        elif mode == 'png':
            files = sorted(glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True))[:8]
            for f in files:
                im = np.array(Image.open(f).convert('L'), dtype=np.float32)/255.0
                imgs.append(np.expand_dims(im,0))
    else:
        imgs = [np.random.rand(1, resize, resize).astype(np.float32) for _ in range(8)]

    imgs_t = torch.from_numpy(np.stack(imgs)).float()
    with torch.no_grad():
        rec, mu, logvar, z = vae(imgs_t.to(device))
        rec = rec.cpu().numpy()
        orig = imgs_t.numpy()

    fig, axs = plt.subplots(2, len(orig), figsize=(len(orig)*2,4))
    for i in range(len(orig)):
        axs[0,i].imshow(orig[i,0], cmap='gray')
        axs[0,i].axis('off')
        axs[1,i].imshow(rec[i,0], cmap='gray')
        axs[1,i].axis('off')
    plt.suptitle("Top: original, Bottom: reconstruction")
    plt.savefig(os.path.join(sample_dir, "recon_grid.png"))
    plt.close()

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='/home/groups/comp3710/', help='data root (npy, png or nifti)')
    p.add_argument('--mode', type=str, default='npy', choices=['npy','png','nifti'])
    p.add_argument('--out-dir', type=str, default='./outputs')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--latent-dim', type=int, default=2)
    p.add_argument('--kld-weight', type=float, default=1.0)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--resize', type=int, default=128)
    p.add_argument('--mode-run', type=str, default='train', choices=['train','demo'])
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--sample-dir', type=str, default='./samples')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode_run == 'train':
        train(args)
    else:
        if args.checkpoint is None:
            raise RuntimeError("Provide --checkpoint path for demo")
        reconstruct_and_save(args.checkpoint, sample_dir=args.sample_dir,
                             device='cuda' if torch.cuda.is_available() else 'cpu',
                             mode=args.mode, data_dir=args.data_dir, resize=args.resize)
