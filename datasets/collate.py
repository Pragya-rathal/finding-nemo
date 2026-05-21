import torch

def fish_collate(batch):
    images=torch.stack([b['image'] for b in batch],0)
    return {'image':images,'boxes':[b['boxes'] for b in batch],'labels':[b['labels'] for b in batch],'order_id':torch.stack([b['order_id'] for b in batch]),'family_id':torch.stack([b['family_id'] for b in batch]),'genus_id':torch.stack([b['genus_id'] for b in batch]),'species_id':torch.stack([b['species_id'] for b in batch]),'trait_vector':torch.stack([b['trait_vector'] for b in batch])}
