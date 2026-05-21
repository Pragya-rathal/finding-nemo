import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(size=640,train=True):
    if train:
        return A.Compose([A.Resize(size,size),A.RandomCrop(size,size,p=0.5),A.HorizontalFlip(p=0.5),A.Blur(p=0.2),A.CLAHE(p=0.2),A.ColorJitter(p=0.3),A.Normalize(),ToTensorV2()],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['labels']))
    return A.Compose([A.Resize(size,size),A.Normalize(),ToTensorV2()],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['labels']))
