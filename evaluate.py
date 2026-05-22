from __future__ import annotations

import argparse
import torch

from utils.config import load_config
from data.transforms import build_transforms
from datasets.fishnet import build_datasets
from models.lhlf_netpp import LHLFNetPP
from losses.unified_loss import UnifiedLoss
from training.build import build_loader
from training.engine import run_epoch
from utils.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    _, va_ds, meta = build_datasets(cfg, build_transforms(cfg['data']['image_size']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LHLFNetPP(cfg, n_species=meta['num_species']).to(device)
    ckpt = load_checkpoint(args.ckpt, model)
    if 'meta' in ckpt and ckpt['meta'].get('num_species') != meta['num_species']:
        model = LHLFNetPP(cfg, n_species=int(ckpt['meta']['num_species'])).to(device)
        load_checkpoint(args.ckpt, model)

    res = run_epoch(
        model,
        build_loader(va_ds, cfg, False),
        UnifiedLoss(cfg),
        optimizer=None,
        scaler=torch.cuda.amp.GradScaler(enabled=False),
        device=device,
        train=False,
        amp=cfg['system']['amp'],
    )
    print(res)


if __name__ == '__main__':
    main()
