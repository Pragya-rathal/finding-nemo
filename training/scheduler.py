import torch

def build_scheduler(opt,cfg,steps_per_epoch):
    t=cfg['train']; total=t['epochs']*steps_per_epoch; warm=t['scheduler']['warmup_epochs']*steps_per_epoch; min_lr=t['scheduler']['min_lr']
    def fn(step):
        if step<warm: return max(1e-8,float(step+1)/max(1,warm))
        p=(step-warm)/max(1,total-warm)
        return min_lr/t['optimizer']['lr']+0.5*(1-min_lr/t['optimizer']['lr'])*(1+torch.cos(torch.tensor(p*3.1415926535))).item()
    return torch.optim.lr_scheduler.LambdaLR(opt,fn)
