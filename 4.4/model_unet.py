import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, features=(64,128,256,512)):
        super().__init__()
        f1,f2,f3,f4 = features
        self.enc1 = conv_block(in_ch,f1)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(f1,f2)
        self.enc3 = conv_block(f2,f3)
        self.enc4 = conv_block(f3,f4)

        self.up3 = nn.ConvTranspose2d(f4,f3,kernel_size=2,stride=2)
        self.dec3 = conv_block(f4,f3)
        self.up2 = nn.ConvTranspose2d(f3,f2,kernel_size=2,stride=2)
        self.dec2 = conv_block(f3,f2)
        self.up1 = nn.ConvTranspose2d(f2,f1,kernel_size=2,stride=2)
        self.dec1 = conv_block(f2,f1)

        self.outc = nn.Conv2d(f1,out_ch,kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.up3(e4)
        d3 = torch.cat([d3,e3],dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2,e2],dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1,e1],dim=1)
        d1 = self.dec1(d1)

        out = self.outc(d1)
        return out
