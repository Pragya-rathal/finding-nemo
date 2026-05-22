from __future__ import annotations
import torch
import torch.nn as nn

class AnatomicalAttention(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c//4, 1), nn.ReLU(), nn.Conv2d(c//4, c, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch = self.channel(x)
        x = x * ch
        mx, _ = x.max(dim=1, keepdim=True)
        av = x.mean(dim=1, keepdim=True)
        sp = self.spatial(torch.cat([mx, av], dim=1))
        return x * sp
