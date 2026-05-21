import torch

def generate_points(h,w,stride,device):
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    return torch.stack([(x+0.5)*stride,(y+0.5)*stride],-1).reshape(-1,2)
