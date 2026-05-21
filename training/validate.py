from __future__ import annotations
import torch
from utils.misc import move_to_device

def validate(model,loader,loss_fn,device):
    model.eval(); total=0.0
    with torch.no_grad():
        for batch in loader:
            batch=move_to_device(batch,device); out=model(batch['image']); total+=loss_fn(out,batch)['loss'].item()
    return {'val/loss':total/max(1,len(loader))}
