from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training.train import run_training
from utils.config import load_config
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LHLF-Net++ training entrypoint")
    parser.add_argument("--mode", choices=["train"], default="train")
    parser.add_argument("--config_dir", default=str(ROOT / "configs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_dir)
    set_seed(cfg.get("seed", 42))
    torch.backends.cudnn.benchmark = bool(cfg.get("system", {}).get("cudnn_benchmark", True))
    if args.mode == "train":
        run_training(cfg)


if __name__ == "__main__":
    main()
