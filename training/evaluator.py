from __future__ import annotations
from training.metrics import cls_metrics

def evaluate(model,loader,device):
    return {'mAP':0.0,'AP50':0.0,'AP75':0.0,'AP_small':0.0,'top1':0.0,'top5':0.0,'macro_f1':0.0,'balanced_acc':0.0}
