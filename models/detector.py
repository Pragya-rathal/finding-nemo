import torch
from torch import nn
import torchvision.ops as ops
from .anchors import generate_points

class DetectHead(nn.Module):
    def __init__(self,in_channels=256,num_classes=300,strides=[8,16,32,64]):
        super().__init__(); self.strides=strides
        self.cls=nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1),nn.GELU(),nn.Conv2d(in_channels,num_classes,1)) for _ in strides])
        self.obj=nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1),nn.GELU(),nn.Conv2d(in_channels,1,1)) for _ in strides])
        self.reg=nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1),nn.GELU(),nn.Conv2d(in_channels,4,1)) for _ in strides])
    def forward(self,pyramid):
        out=[]
        for i,p in enumerate(pyramid):
            out.append({'cls':self.cls[i](p),'obj':self.obj[i](p),'reg':self.reg[i](p),'stride':self.strides[i]})
        return out
    def decode(self,preds,score_thr=0.05,nms_thr=0.6,topk=300):
        dets=[]
        for b in range(preds[0]['cls'].shape[0]):
            bs=[]; ss=[]; ls=[]
            for p in preds:
                cls=p['cls'][b].sigmoid(); obj=p['obj'][b].sigmoid(); reg=p['reg'][b]
                c,h,w=cls.shape
                points=generate_points(h,w,p['stride'],cls.device)
                cls=cls.reshape(c,-1).permute(1,0); obj=obj.reshape(-1,1)
                score=(cls*obj)
                vals,lab=score.max(1)
                keep=vals>score_thr
                if keep.sum()==0: continue
                pts=points[keep]; r=reg.reshape(4,-1).permute(1,0)[keep]
                x1=pts[:,0]-r[:,0].relu(); y1=pts[:,1]-r[:,1].relu(); x2=pts[:,0]+r[:,2].relu(); y2=pts[:,1]+r[:,3].relu()
                bs.append(torch.stack([x1,y1,x2,y2],1)); ss.append(vals[keep]); ls.append(lab[keep])
            if not bs:
                dets.append({'boxes':torch.zeros((0,4)), 'scores':torch.zeros(0), 'labels':torch.zeros(0,dtype=torch.long)}); continue
            bxs=torch.cat(bs); sc=torch.cat(ss); lb=torch.cat(ls)
            idx=ops.nms(bxs,sc,nms_thr)[:topk]
            dets.append({'boxes':bxs[idx], 'scores':sc[idx], 'labels':lb[idx]})
        return dets
