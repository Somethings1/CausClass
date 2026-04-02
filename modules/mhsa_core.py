import torch
import torch.nn as nn
import ultralytics.nn.modules as modules

class MHSA(nn.Module):
    def __init__(self, c1, c2, h=4):
        super().__init__()
        self.h = h
        self.dh = c2 // h
        self.qkv = nn.Linear(c1, c2 * 3)
        self.project = nn.Linear(c2, c2)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        qkv = self.qkv(x_flat).view(B, -1, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.project(out).permute(0, 2, 1).view(B, C, H, W)
        return out

def patch_yolo_mhsa():
    """Hàm này dùng để tiêm MHSA vào thư viện Ultralytics"""
    setattr(modules, 'MHSA', MHSA)
