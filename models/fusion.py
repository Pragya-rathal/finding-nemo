import torch
from torch import nn
import torch.nn.functional as F

class MultiScaleFusion(nn.Module):
    def __init__(self,in_channels,out_channels=256):
        super().__init__()
        self.proj=nn.ModuleList([nn.Conv2d(c,out_channels,1) for c in in_channels])
        self.refine=nn.Sequential(nn.Conv2d(out_channels*4,out_channels,3,1,1),nn.BatchNorm2d(out_channels),nn.GELU(),nn.Conv2d(out_channels,out_channels,3,1,1),nn.BatchNorm2d(out_channels),nn.GELU())
    def forward(self,feats):
        size=feats[0].shape[-2:]
        xs=[]
        for i,f in enumerate(feats):
            p=self.proj[i](f)
            if p.shape[-2:]!=size: p=F.interpolate(p,size=size,mode='bilinear',align_corners=False)
            xs.append(p)
        return self.refine(torch.cat(xs,1))
