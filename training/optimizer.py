import torch

def build_optimizer(model,cfg):
    o=cfg['train']['optimizer']
    return torch.optim.AdamW(model.parameters(),lr=o['lr'],weight_decay=o['weight_decay'],betas=tuple(o['betas']))
