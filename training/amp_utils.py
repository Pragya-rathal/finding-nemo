import torch

def get_scaler(enabled=True):
    return torch.cuda.amp.GradScaler(enabled=enabled)
