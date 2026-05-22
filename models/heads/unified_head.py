from __future__ import annotations
import torch
import torch.nn as nn

class UnifiedHead(nn.Module):
    def __init__(self, c: int, embed_dim: int, n_order: int, n_family: int, n_genus: int, n_species: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.obj = nn.Linear(c, 1)
        self.box = nn.Linear(c, 4)
        self.embed = nn.Linear(c, embed_dim)
        self.order = nn.Linear(embed_dim, n_order)
        self.family = nn.Linear(embed_dim, n_family)
        self.genus = nn.Linear(embed_dim, n_genus)
        self.species = nn.Linear(embed_dim, n_species)

    def forward(self, feats: list[torch.Tensor]):
        x = torch.stack([f.mean(dim=[2,3]) for f in feats], dim=0).mean(dim=0)
        emb = self.embed(x)
        return {
            'objectness': self.obj(x).squeeze(-1),
            'bbox': self.box(x),
            'embedding': emb,
            'order_logits': self.order(emb),
            'family_logits': self.family(emb),
            'genus_logits': self.genus(emb),
            'species_logits': self.species(emb),
        }
