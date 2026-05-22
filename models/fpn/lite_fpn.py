from __future__ import annotations
import torch.nn as nn

class LiteFPN(nn.Module):
    def __init__(self, in_channels, out_channels=160):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.outs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])

    def forward(self, feats):
        xs = [l(f) for l, f in zip(self.lats, feats)]
        for i in range(len(xs)-2, -1, -1):
            xs[i] = xs[i] + nn.functional.interpolate(xs[i+1], size=xs[i].shape[-2:], mode='nearest')
        return [o(x) for o, x in zip(self.outs, xs)]
