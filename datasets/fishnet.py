from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

@dataclass
class Sample:
    path: str
    species: str
    order: str
    family: str
    genus: str


def discover_fishnet_root(user_root: str | None = None) -> Path:
    cands = []
    if user_root and user_root != 'auto':
        cands.append(Path(user_root))
    cands.extend([Path('/kaggle/input/datasets'), Path('/kaggle/input')])
    for base in cands:
        if not base.exists():
            continue
        for p in base.rglob('fishnet'):
            if p.is_dir():
                return p
    raise FileNotFoundError('Unable to discover fishnet dataset root.')


def parse_taxonomy(species_name: str) -> tuple[str, str, str]:
    parts = re.split(r'[_\- ]+', species_name.strip())
    genus = parts[0].lower() if parts else 'unknown_genus'
    family = (parts[1].lower() + '_fam') if len(parts) > 1 else 'unknown_family'
    order = (parts[2].lower() + '_ord') if len(parts) > 2 else 'unknown_order'
    return order, family, genus


def scan_samples(root: Path) -> list[Sample]:
    out: list[Sample] = []
    for species_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        species = species_dir.name
        order, family, genus = parse_taxonomy(species)
        for f in species_dir.rglob('*'):
            if f.suffix.lower() in IMG_EXT:
                out.append(Sample(str(f), species, order, family, genus))
    return out


def safe_read(path: str) -> np.ndarray | None:
    try:
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))[:, :, ::-1]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

class FishNetDataset(Dataset):
    def __init__(self, samples: list[Sample], mappings: dict[str, dict[str, int]], transform=None):
        self.samples = samples
        self.m = mappings
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = safe_read(s.path)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        h, w = img.shape[-2:]
        bbox = torch.tensor([0.0, 0.0, float(w), float(h)], dtype=torch.float32)
        target = {
            'species': torch.tensor(self.m['species'][s.species], dtype=torch.long),
            'genus': torch.tensor(self.m['genus'][s.genus], dtype=torch.long),
            'family': torch.tensor(self.m['family'][s.family], dtype=torch.long),
            'order': torch.tensor(self.m['order'][s.order], dtype=torch.long),
            'bbox': bbox,
            'objectness': torch.tensor(1.0, dtype=torch.float32),
        }
        return img, target

def build_datasets(cfg: dict, transforms: dict[str, Any]):
    root = discover_fishnet_root(cfg['data'].get('dataset_root'))
    samples = scan_samples(root)
    if cfg['data'].get('max_samples'):
        samples = samples[: int(cfg['data']['max_samples'])]
    species = sorted({s.species for s in samples})
    genus = sorted({s.genus for s in samples})
    family = sorted({s.family for s in samples})
    order = sorted({s.order for s in samples})
    mappings = {k: {v: i for i, v in enumerate(vals)} for k, vals in [('species', species), ('genus', genus), ('family', family), ('order', order)]}
    idx = list(range(len(samples)))
    tr, va = train_test_split(idx, test_size=cfg['data'].get('val_split', 0.2), random_state=cfg.get('seed', 42), stratify=[samples[i].species for i in idx] if len(species) > 1 else None)
    train_ds = FishNetDataset([samples[i] for i in tr], mappings, transforms['train'])
    val_ds = FishNetDataset([samples[i] for i in va], mappings, transforms['val'])
    meta = {'root': str(root), 'num_samples': len(samples), 'num_species': len(species), 'num_genus': len(genus), 'num_family': len(family), 'num_order': len(order), 'mappings': mappings}
    return train_ds, val_ds, meta
