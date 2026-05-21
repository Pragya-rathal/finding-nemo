from __future__ import annotations
import torch
from torch import nn

class DropPath(nn.Module):
    def __init__(self, p: float=0.0):
        super().__init__(); self.p=p
    def forward(self, x):
        if self.p==0.0 or not self.training: return x
        keep=1-self.p
        shape=(x.shape[0],)+(1,)*(x.ndim-1)
        rnd=keep+torch.rand(shape, device=x.device)
        rnd.floor_()
        return x/keep*rnd

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim:int, heads:int=8, sr_ratio:int=1, attn_drop:float=0.0, proj_drop:float=0.0):
        super().__init__(); self.heads=heads; self.dim=dim; self.scale=(dim//heads)**-0.5
        self.q=nn.Linear(dim,dim); self.kv=nn.Linear(dim,dim*2)
        self.attn_drop=nn.Dropout(attn_drop); self.proj=nn.Linear(dim,dim); self.proj_drop=nn.Dropout(proj_drop)
        self.sr_ratio=sr_ratio
        if sr_ratio>1:
            self.sr=nn.Conv2d(dim,dim,sr_ratio,sr_ratio)
            self.norm=nn.LayerNorm(dim)
    def forward(self,x,h,w):
        b,n,c=x.shape
        q=self.q(x).reshape(b,n,self.heads,c//self.heads).permute(0,2,1,3)
        if self.sr_ratio>1:
            xs=x.permute(0,2,1).reshape(b,c,h,w)
            xs=self.sr(xs).reshape(b,c,-1).permute(0,2,1)
            xs=self.norm(xs)
            kv=self.kv(xs)
        else:
            kv=self.kv(x)
        kv=kv.reshape(b,-1,2,self.heads,c//self.heads).permute(2,0,3,1,4)
        k,v=kv[0],kv[1]
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=self.attn_drop(attn.softmax(-1))
        x=(attn@v).transpose(1,2).reshape(b,n,c)
        return self.proj_drop(self.proj(x))

class MixFFN(nn.Module):
    def __init__(self, dim:int, ratio:float=4.0, drop:float=0.0):
        super().__init__(); hidden=int(dim*ratio)
        self.fc1=nn.Linear(dim,hidden); self.dwconv=nn.Conv2d(hidden,hidden,3,1,1,groups=hidden)
        self.act=nn.GELU(); self.fc2=nn.Linear(hidden,dim); self.drop=nn.Dropout(drop)
    def forward(self,x,h,w):
        b,n,c=x.shape
        x=self.fc1(x)
        x=self.act(x)
        x=x.transpose(1,2).reshape(b,-1,h,w)
        x=self.dwconv(x)
        x=x.flatten(2).transpose(1,2)
        x=self.drop(self.act(x)); x=self.fc2(x); return self.drop(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim:int, heads:int, sr_ratio:int, mlp_ratio:float, drop:float=0.0, attn_drop:float=0.0, dp:float=0.0):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim)
        self.attn=EfficientSelfAttention(dim,heads,sr_ratio,attn_drop,drop)
        self.ffn=MixFFN(dim,mlp_ratio,drop)
        self.dp=DropPath(dp)
    def forward(self,x,h,w):
        x=x+self.dp(self.attn(self.n1(x),h,w))
        x=x+self.dp(self.ffn(self.n2(x),h,w))
        return x
