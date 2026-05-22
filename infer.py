from __future__ import annotations

import argparse
import cv2
import torch

from data.transforms import build_transforms
from models.lhlf_netpp import LHLFNetPP
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f'Unable to read image: {args.image}')

    # bootstrap with fallback species count, then load checkpoint/meta
    model = LHLFNetPP(cfg, n_species=cfg['model'].get('num_species', 500)).to(device).eval()
    ckpt = load_checkpoint(args.ckpt, model)

    if 'meta' in ckpt and 'num_species' in ckpt['meta']:
        n_species = int(ckpt['meta']['num_species'])
        if model.head.species.out_features != n_species:
            model = LHLFNetPP(cfg, n_species=n_species).to(device).eval()
            load_checkpoint(args.ckpt, model)

    t = build_transforms(cfg['data']['image_size'])['val']
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = t(image=img)['image'].unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad(), torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=cfg['system']['amp']):
        out = model(x)

    p = torch.softmax(out['species_logits'], dim=-1)
    conf, cls = p.max(dim=-1)
    if conf.item() < cfg['inference']['fallback_taxonomy_threshold']:
        print({
            'mode': 'taxonomy_fallback',
            'order': out['order_logits'].argmax(-1).item(),
            'family': out['family_logits'].argmax(-1).item(),
            'genus': out['genus_logits'].argmax(-1).item(),
            'species_confidence': conf.item(),
        })
    else:
        print({
            'mode': 'species',
            'species_id': cls.item(),
            'confidence': conf.item(),
            'bbox': out['bbox'][0].tolist(),
        })


if __name__ == '__main__':
    main()
