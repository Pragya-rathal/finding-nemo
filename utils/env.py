from __future__ import annotations
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_torch(cfg: dict) -> None:
    system = cfg.get('system', {})
    torch.backends.cudnn.benchmark = bool(system.get('cudnn_benchmark', True))
    torch.backends.cuda.matmul.allow_tf32 = bool(system.get('allow_tf32', True))
