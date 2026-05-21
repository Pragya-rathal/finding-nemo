from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.fishnet_dataset import FishNetDataset
from datasets.transforms import build_transforms
from datasets.collate import fish_collate
from models.lhlfnetpp import LHLFNetPP
from models.losses import UnifiedLoss
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.amp_utils import get_scaler
from training.engine import train_one_epoch
from training.validate import validate
from utils.checkpoint import save_checkpoint
from utils.device import get_device

def run_training(cfg):
    device=get_device()
    tr=FishNetDataset(cfg['dataset']['train_images'],cfg['dataset']['train_annotations'],build_transforms(cfg['model']['image_size'],True))
    va=FishNetDataset(cfg['dataset']['val_images'],cfg['dataset']['val_annotations'],build_transforms(cfg['model']['image_size'],False))
    trl=DataLoader(tr,batch_size=cfg['train']['batch_size'],shuffle=True,num_workers=cfg['system']['num_workers'],pin_memory=cfg['system']['pin_memory'],collate_fn=fish_collate)
    val=DataLoader(va,batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=cfg['system']['num_workers'],pin_memory=cfg['system']['pin_memory'],collate_fn=fish_collate)
    model=LHLFNetPP(cfg).to(device); loss_fn=UnifiedLoss(cfg)
    opt=build_optimizer(model,cfg); sch=build_scheduler(opt,cfg,len(trl)); scaler=get_scaler(cfg['train']['amp']); writer=SummaryWriter(cfg['output_dir'])
    best=1e9; patience=0
    for e in range(cfg['train']['epochs']):
        tl=train_one_epoch(model,trl,opt,sch,loss_fn,scaler,device,e,cfg['train']['accumulation_steps'],writer)
        vm=validate(model,val,loss_fn,device); writer.add_scalar('val/loss',vm['val/loss'],e)
        if vm['val/loss']<best:
            best=vm['val/loss']; patience=0
            save_checkpoint(f"{cfg['output_dir']}/best.pt",{'model':model.state_dict(),'epoch':e,'best':best})
        else:
            patience+=1
        if patience>=cfg['train']['early_stopping']['patience']: break
    writer.close()
