import torch

def box_iou(boxes1,boxes2):
    area1=(boxes1[:,2]-boxes1[:,0]).clamp(0)*(boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2=(boxes2[:,2]-boxes2[:,0]).clamp(0)*(boxes2[:,3]-boxes2[:,1]).clamp(0)
    lt=torch.max(boxes1[:,None,:2],boxes2[:,:2]); rb=torch.min(boxes1[:,None,2:],boxes2[:,2:])
    wh=(rb-lt).clamp(0); inter=wh[:,:,0]*wh[:,:,1]
    return inter/(area1[:,None]+area2-inter+1e-6)
