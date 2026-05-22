from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from utils.config import load_config
from utils.env import set_seed, setup_torch
from utils.io import save_json
from utils.checkpoint import save_checkpoint
from data.transforms import build_transforms
from datasets.fishnet import build_datasets
from models.lhlf_netpp import LHLFNetPP
from losses.unified_loss import UnifiedLoss
from training.build import build_loader
from training.engine import run_epoch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42)); setup_torch(cfg)
    tfm = build_transforms(cfg['data']['image_size'])
    tr_ds, va_ds, meta = build_datasets(cfg, tfm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LHLFNetPP(cfg, n_species=meta['num_species']).to(device)
    if cfg['system'].get('channels_last', True): model = model.to(memory_format=torch.channels_last)
    opt = AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scaler = GradScaler(enabled=cfg['system']['amp'] and device.type == 'cuda')
    crit = UnifiedLoss(cfg)
    tr_ld = build_loader(tr_ds, cfg, True); va_ld = build_loader(va_ds, cfg, False)
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True); save_json(meta, out/'dataset_meta.json')
    best = -1.0
    for ep in range(cfg['train']['epochs']):
        tr = run_epoch(model, tr_ld, crit, opt, scaler, device, True, cfg['system']['amp'])
        va = run_epoch(model, va_ld, crit, opt, scaler, device, False, cfg['system']['amp'])
        print(f"epoch={ep+1} train={tr} val={va}")
        save_checkpoint({'epoch': ep+1, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'meta': meta}, out, 'last.ckpt')
        if va['species_acc'] > best:
            best = va['species_acc']; save_checkpoint({'epoch': ep+1, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'meta': meta}, out, 'best.ckpt')

if __name__ == '__main__':
    main()
