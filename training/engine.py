from __future__ import annotations
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.misc import move_to_device

def train_one_epoch(model,loader,optimizer,scheduler,loss_fn,scaler,device,epoch,accum_steps=1,writer:SummaryWriter|None=None):
    model.train(); total=0.0
    optimizer.zero_grad(set_to_none=True)
    pbar=tqdm(loader,desc=f'train {epoch}')
    for it,batch in enumerate(pbar):
        batch=move_to_device(batch,device)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out=model(batch['image']); ldict=loss_fn(out,batch); loss=ldict['loss']/accum_steps
        scaler.scale(loss).backward()
        if (it+1)%accum_steps==0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        total+=loss.item()*accum_steps; pbar.set_postfix(loss=total/(it+1))
    if writer: writer.add_scalar('train/loss',total/max(1,len(loader)),epoch)
    return total/max(1,len(loader))
