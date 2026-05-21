import torch
from torch import nn

class TaxonomyHeads(nn.Module):
    def __init__(self,dim:int,order:int,family:int,genus:int,species:int,traits:int):
        super().__init__()
        self.order=nn.Linear(dim,order); self.family=nn.Linear(dim,family); self.genus=nn.Linear(dim,genus); self.species=nn.Linear(dim,species); self.trait=nn.Linear(dim,traits)
    def forward(self,x):
        return {'order_logits':self.order(x),'family_logits':self.family(x),'genus_logits':self.genus(x),'species_logits':self.species(x),'trait_logits':self.trait(x)}
