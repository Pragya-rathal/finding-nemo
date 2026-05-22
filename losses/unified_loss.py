from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        l = cfg['loss']
        self.tax_w = l.get('taxonomy_weight', 1.0)
        self.bbox_w = l.get('bbox_weight', 2.0)
        self.obj_w = l.get('objectness_weight', 1.0)
        self.smooth = l.get('label_smoothing', 0.0)

    def forward(self, out, target):
        ce = 0.0
        ce += F.cross_entropy(out['species_logits'], target['species'], label_smoothing=self.smooth)
        ce += F.cross_entropy(out['genus_logits'], target['genus'], label_smoothing=self.smooth)
        ce += F.cross_entropy(out['family_logits'], target['family'], label_smoothing=self.smooth)
        ce += F.cross_entropy(out['order_logits'], target['order'], label_smoothing=self.smooth)
        bbox = F.smooth_l1_loss(out['bbox'], target['bbox'])
        obj = F.binary_cross_entropy_with_logits(out['objectness'], target['objectness'])
        total = self.tax_w * ce + self.bbox_w * bbox + self.obj_w * obj
        return total, {'taxonomy': ce.item(), 'bbox': bbox.item(), 'objectness': obj.item(), 'total': total.item()}
