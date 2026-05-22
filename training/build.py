from __future__ import annotations
from torch.utils.data import DataLoader

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return __import__('torch').stack(list(imgs), dim=0), list(targets)

def build_loader(ds, cfg, train=True):
    nw = int(cfg['system']['num_workers'])
    return DataLoader(
        ds,
        batch_size=int(cfg['data']['batch_size']),
        shuffle=train,
        num_workers=nw,
        pin_memory=bool(cfg['system']['pin_memory']),
        persistent_workers=bool(cfg['system']['persistent_workers']) if nw > 0 else False,
        collate_fn=collate_fn,
    )
