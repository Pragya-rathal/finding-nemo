from __future__ import annotations
import argparse, torch
from models.lhlf_netpp import LHLFNetPP
from utils.config import load_config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/default.yaml'); ap.add_argument('--species', type=int, default=500); ap.add_argument('--out', default='lhlfnetpp.onnx')
    a=ap.parse_args(); cfg=load_config(a.config)
    m=LHLFNetPP(cfg, n_species=a.species).eval()
    x=torch.randn(1,3,cfg['data']['image_size'],cfg['data']['image_size'])
    torch.onnx.export(m, x, a.out, input_names=['images'], output_names=['output'], opset_version=17)
if __name__ == '__main__': main()
