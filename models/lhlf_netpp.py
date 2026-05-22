from __future__ import annotations
import torch.nn as nn
from models.backbones.mit_encoder import MiTEncoder
from models.fpn.lite_fpn import LiteFPN
from models.attention.anatomical_attention import AnatomicalAttention
from models.heads.unified_head import UnifiedHead

class LHLFNetPP(nn.Module):
    def __init__(self, cfg: dict, n_species: int):
        super().__init__()
        self.encoder = MiTEncoder(cfg['model']['backbone'], pretrained=False)
        self.fpn = LiteFPN(self.encoder.channels, cfg['model']['fpn_out_channels'])
        self.attn = nn.ModuleList([AnatomicalAttention(cfg['model']['fpn_out_channels']) for _ in self.encoder.channels])
        self.head = UnifiedHead(
            cfg['model']['fpn_out_channels'],
            cfg['model']['embed_dim'],
            cfg['model']['num_orders'],
            cfg['model']['num_families'],
            cfg['model']['num_genera'],
            n_species,
        )

    def forward(self, x):
        feats = self.encoder(x)
        feats = self.fpn(feats)
        feats = [a(f) for a, f in zip(self.attn, feats)]
        return self.head(feats)
