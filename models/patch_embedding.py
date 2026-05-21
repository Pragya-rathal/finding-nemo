from __future__ import annotations
import torch
from torch import nn

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int=7, stride: int=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1,2).reshape(b,c,h,w)
        return x
