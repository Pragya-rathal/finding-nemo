from __future__ import annotations
import torch
from pathlib import Path

def save_checkpoint(state: dict, out_dir: str, name: str = 'last.ckpt'):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(state, p / name)

def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt
