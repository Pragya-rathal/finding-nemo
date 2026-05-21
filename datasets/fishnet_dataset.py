from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import cv2, torch, numpy as np
from torch.utils.data import Dataset
from .coco_parser import load_coco

class FishNetDataset(Dataset):
    def __init__(self,image_dir:str,ann_path:str,transforms=None):
        self.image_dir=Path(image_dir); self.coco=load_coco(ann_path); self.transforms=transforms
        self.images=self.coco['images']
        self.ann_by_img={}
        for a in self.coco['annotations']: self.ann_by_img.setdefault(a['image_id'],[]).append(a)
    def __len__(self): return len(self.images)
    def __getitem__(self,idx:int)->Dict[str,Any]:
        im=self.images[idx]; img=cv2.cvtColor(cv2.imread(str(self.image_dir/im['file_name'])),cv2.COLOR_BGR2RGB)
        anns=self.ann_by_img.get(im['id'],[])
        boxes=[]; labels=[]
        for a in anns:
            x,y,w,h=a['bbox']; boxes.append([x,y,x+w,y+h]); labels.append(a['category_id'])
        if not boxes: boxes=[[0,0,1,1]]; labels=[0]
        order_id=anns[0].get('order_id',0) if anns else 0
        family_id=anns[0].get('family_id',0) if anns else 0
        genus_id=anns[0].get('genus_id',0) if anns else 0
        species_id=anns[0].get('species_id',0) if anns else 0
        trait_vector=np.array(anns[0].get('trait_vector',[0]*32) if anns else [0]*32,dtype=np.float32)
        if self.transforms:
            t=self.transforms(image=img,bboxes=boxes,labels=labels); img=t['image']; boxes=t['bboxes']; labels=t['labels']
        return {'image':img,'boxes':torch.tensor(boxes,dtype=torch.float32),'labels':torch.tensor(labels,dtype=torch.long),'order_id':torch.tensor(order_id,dtype=torch.long),'family_id':torch.tensor(family_id,dtype=torch.long),'genus_id':torch.tensor(genus_id,dtype=torch.long),'species_id':torch.tensor(species_id,dtype=torch.long),'trait_vector':torch.tensor(trait_vector,dtype=torch.float32)}
