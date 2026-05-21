import torch
from torch import nn
from .encoder import HierarchicalTransformerEncoder
from .fusion import MultiScaleFusion
from .fpn import FPN
from .attention import AnatomicalAttention
from .detector import DetectHead
from .embedding import SharedEmbedding
from .taxonomy_heads import TaxonomyHeads

class LHLFNetPP(nn.Module):
    def __init__(self,cfg):
        super().__init__(); m=cfg['model']
        self.encoder=HierarchicalTransformerEncoder(in_chans=m['in_channels'],**m['encoder'])
        self.fusion=MultiScaleFusion(m['encoder']['embed_dims'],m['fusion']['out_channels'])
        self.fpn=FPN(m['encoder']['embed_dims'],m['fpn']['out_channels'])
        self.attn=AnatomicalAttention(m['fusion']['out_channels'],m['attention']['kernel_size'])
        self.det=DetectHead(m['fpn']['out_channels'],m['detector']['num_classes'],m['detector']['strides'])
        self.emb=SharedEmbedding(m['fusion']['out_channels'],m['embedding']['dim'],m['embedding']['dropout'])
        t=m['taxonomy']; self.tax=TaxonomyHeads(m['embedding']['dim'],t['order_classes'],t['family_classes'],t['genus_classes'],t['species_classes'],t['trait_classes'])
    def forward(self,x):
        feats=self.encoder(x)
        fused=self.fusion(feats)
        refined,attn=self.attn(fused)
        pyramids=self.fpn(feats)
        detections=self.det(pyramids)
        emb=self.emb(refined)
        out=self.tax(emb)
        out.update({'detections':detections,'attention_map':attn,'shared_embedding':emb,'det_losses':torch.tensor(0.0,device=x.device)})
        return out
