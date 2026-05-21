from __future__ import annotations
from typing import Dict
import torch

def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out={}
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k]=v.to(device, non_blocking=True)
        elif isinstance(v, list):
            out[k]=[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in v]
        else:
            out[k]=v
    return out
