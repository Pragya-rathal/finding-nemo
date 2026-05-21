import torch
from torch import nn

class SharedEmbedding(nn.Module):
    def __init__(self,in_channels:int,dim:int=512,dropout:float=0.1):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.proj=nn.Sequential(nn.Linear(in_channels,dim),nn.LayerNorm(dim),nn.GELU(),nn.Dropout(dropout))
    def forward(self,x):
        x=self.pool(x).flatten(1)
        return self.proj(x)
