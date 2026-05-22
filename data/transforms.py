from __future__ import annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(size: int = 512):
    train = A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])
    val = A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ])
    return {'train': train, 'val': val}
