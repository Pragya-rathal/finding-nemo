from __future__ import annotations
import torch
from torch import nn
from .patch_embedding import OverlapPatchEmbed
from .transformer_blocks import TransformerBlock

class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self,in_chans=3,embed_dims=[64,128,256,512],depths=[2,2,4,2],num_heads=[1,2,4,8],sr_ratios=[8,4,2,1],mlp_ratios=[4,4,4,4],drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.1):
        super().__init__()
        self.patch_embeds=nn.ModuleList([
            OverlapPatchEmbed(in_chans,embed_dims[0],7,4),
            OverlapPatchEmbed(embed_dims[0],embed_dims[1],3,2),
            OverlapPatchEmbed(embed_dims[1],embed_dims[2],3,2),
            OverlapPatchEmbed(embed_dims[2],embed_dims[3],3,2),
        ])
        dpr=torch.linspace(0,drop_path_rate,sum(depths)).tolist(); idx=0; self.stages=nn.ModuleList()
        for i,d in enumerate(depths):
            blk=nn.ModuleList([TransformerBlock(embed_dims[i],num_heads[i],sr_ratios[i],mlp_ratios[i],drop_rate,attn_drop_rate,dpr[idx+j]) for j in range(d)])
            idx+=d; self.stages.append(blk)
    def forward(self,x):
        feats=[]
        for pe,blocks in zip(self.patch_embeds,self.stages):
            x=pe(x)
            b,c,h,w=x.shape
            t=x.flatten(2).transpose(1,2)
            for blk in blocks: t=blk(t,h,w)
            x=t.transpose(1,2).reshape(b,c,h,w)
            feats.append(x)
        return feats
