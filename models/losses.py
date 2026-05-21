import torch
from torch import nn
import torch.nn.functional as F

class CBFocalLoss(nn.Module):
    def __init__(self,class_counts,beta=0.9999,gamma=2.0):
        super().__init__()
        counts=torch.tensor(class_counts,dtype=torch.float32)
        eff=1.0-torch.pow(beta,counts)
        w=(1.0-beta)/(eff+1e-8)
        self.register_buffer('w',w/w.sum()*len(class_counts)); self.gamma=gamma
    def forward(self,logits,target):
        ce=F.cross_entropy(logits,target,reduction='none',weight=self.w)
        pt=torch.exp(-ce)
        return ((1-pt)**self.gamma*ce).mean()

class UnifiedLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__(); L=cfg['train']['loss']; m=cfg['model']['taxonomy']
        counts=[1]*cfg['model']['detector']['num_classes']
        self.det_cls=CBFocalLoss(counts,L['cb_beta'],L['focal_gamma'])
        self.order=CBFocalLoss([1]*m['order_classes'],L['cb_beta'],L['focal_gamma'])
        self.family=CBFocalLoss([1]*m['family_classes'],L['cb_beta'],L['focal_gamma'])
        self.genus=CBFocalLoss([1]*m['genus_classes'],L['cb_beta'],L['focal_gamma'])
        self.species=CBFocalLoss([1]*m['species_classes'],L['cb_beta'],L['focal_gamma'])
        self.l1=L
    def forward(self,outputs,batch):
        det=outputs['det_losses']
        tax=self.order(outputs['order_logits'],batch['order_id'])+self.family(outputs['family_logits'],batch['family_id'])+self.genus(outputs['genus_logits'],batch['genus_id'])+self.species(outputs['species_logits'],batch['species_id'])+F.binary_cross_entropy_with_logits(outputs['trait_logits'],batch['trait_vector'].float())
        total=det+self.l1['lambda_order']*self.order(outputs['order_logits'],batch['order_id'])+self.l1['lambda_family']*self.family(outputs['family_logits'],batch['family_id'])+self.l1['lambda_genus']*self.genus(outputs['genus_logits'],batch['genus_id'])+self.l1['lambda_species']*self.species(outputs['species_logits'],batch['species_id'])+self.l1['lambda_trait']*F.binary_cross_entropy_with_logits(outputs['trait_logits'],batch['trait_vector'].float())
        return {'loss':total,'taxonomy_loss':tax,'det_loss':det}
