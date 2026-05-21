import torch

def simple_match(pred_boxes,target_boxes):
    if pred_boxes.numel()==0 or target_boxes.numel()==0: return torch.empty(0,dtype=torch.long)
    ious=torch.cdist(pred_boxes,target_boxes)
    return ious.argmin(dim=1)
