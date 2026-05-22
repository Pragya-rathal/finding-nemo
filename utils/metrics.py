from __future__ import annotations
import torch

def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    if logits.numel() == 0:
        return 0.0
    pred = logits.topk(k=min(k, logits.size(-1)), dim=-1).indices
    ok = (pred == targets.unsqueeze(-1)).any(dim=-1).float()
    return ok.mean().item()
