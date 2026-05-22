from __future__ import annotations
import timm
import torch.nn as nn

class MiTEncoder(nn.Module):
    def __init__(self, name='mit_b0', pretrained=False):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, features_only=True)
        self.channels = self.net.feature_info.channels()

    def forward(self, x):
        return self.net(x)
