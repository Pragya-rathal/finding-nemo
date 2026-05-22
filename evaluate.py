from __future__ import annotations
import argparse, torch
from utils.config import load_config
from data.transforms import build_transforms
from datasets.fishnet import build_datasets
from models.lhlf_netpp import LHLFNetPP
from losses.unified_loss import UnifiedLoss
from training.build import build_loader
from training.engine import run_epoch
from utils.checkpoint import load_checkpoint

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/default.yaml'); ap.add_argument('--ckpt', required=True)
    args = ap.parse_args(); cfg = load_config(args.config)
    _, va_ds, meta = build_datasets(cfg, build_transforms(cfg['data']['image_size']))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LHLFNetPP(cfg, n_species=meta['num_species']).to(dev)
    load_checkpoint(args.ckpt, model)
    res = run_epoch(model, build_loader(va_ds, cfg, False), UnifiedLoss(cfg), None, torch.cuda.amp.GradScaler(enabled=False), dev, False, cfg['system']['amp'])
    print(res)
if __name__ == '__main__': main()
