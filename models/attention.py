import torch
from torch import nn

class AnatomicalAttention(nn.Module):
    def __init__(self,channels:int,kernel_size:int=3):
        super().__init__(); p=kernel_size//2
        self.conv=nn.Sequential(nn.Conv2d(channels,channels//2,1),nn.GELU(),nn.Conv2d(channels//2,1,kernel_size,1,p))
    def forward(self,x):
        a=torch.sigmoid(self.conv(x))
        return x*a,a
