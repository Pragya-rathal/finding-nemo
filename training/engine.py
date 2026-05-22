from __future__ import annotations
import torch
from tqdm import tqdm
from utils.metrics import topk_accuracy

def stack_targets(targets, device):
    out = {}
    for k in targets[0].keys():
        out[k] = torch.stack([t[k] for t in targets], dim=0).to(device, non_blocking=True)
    return out

def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True, amp=True):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for imgs, targets in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        t = stack_targets(targets, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=amp):
            out = model(imgs)
            loss, _ = criterion(out, t)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc += topk_accuracy(out['species_logits'].detach(), t['species'], k=1) * bs
        n += bs
    return {'loss': total_loss / max(1, n), 'species_acc': total_acc / max(1, n)}
