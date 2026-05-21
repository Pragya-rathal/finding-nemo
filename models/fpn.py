import torch
from torch import nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self,in_channels,out_channels=256):
        super().__init__()
        self.lateral=nn.ModuleList([nn.Conv2d(c,out_channels,1) for c in in_channels])
        self.outs=nn.ModuleList([nn.Conv2d(out_channels,out_channels,3,1,1) for _ in in_channels])
    def forward(self,feats):
        lat=[l(f) for l,f in zip(self.lateral,feats)]
        for i in range(len(lat)-1,0,-1):
            lat[i-1]=lat[i-1]+F.interpolate(lat[i],size=lat[i-1].shape[-2:],mode='nearest')
        return [o(l) for o,l in zip(self.outs,lat)]
