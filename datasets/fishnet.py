from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


@dataclass
class Sample:
    path: str
    species: str
    order: str
    family: str
    genus: str


def discover_fishnet_root(user_root: str | None = None) -> Path:
    """Discover fishnet root robustly across Kaggle layouts."""
    candidates: list[Path] = []
    if user_root and user_root != 'auto':
        candidates.append(Path(user_root))

    candidates.extend([
        Path('/kaggle/input/datasets'),
        Path('/kaggle/input'),
        Path.cwd(),
    ])

    for base in candidates:
        if not base.exists():
            continue
        if base.is_dir() and base.name.lower() == 'fishnet':
            return base
        for p in base.rglob('*'):
            if p.is_dir() and p.name.lower() == 'fishnet':
                return p

    raise FileNotFoundError(
        'Unable to discover fishnet dataset root. Expected something like '
        '/kaggle/input/datasets/<random>/fishnet or pass data.dataset_root explicitly.'
    )


def parse_taxonomy(species_name: str) -> tuple[str, str, str]:
    parts = [p for p in re.split(r'[_\- ]+', species_name.strip()) if p]
    genus = parts[0].lower() if parts else 'unknown_genus'
    family = f"{parts[1].lower()}_fam" if len(parts) > 1 else 'unknown_family'
    order = f"{parts[2].lower()}_ord" if len(parts) > 2 else 'unknown_order'
    return order, family, genus


def scan_samples(root: Path) -> list[Sample]:
    out: list[Sample] = []
    species_dirs = [p for p in root.iterdir() if p.is_dir()]
    for species_dir in sorted(species_dirs):
        species = species_dir.name
        order, family, genus = parse_taxonomy(species)
        for f in species_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
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
    def __init__(self, samples: list[Sample], mappings: dict[str, dict[str, int]], image_size: int, transform=None):
        self.samples = samples
        self.mappings = mappings
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = safe_read(s.path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        h, w = img.shape[-2:]
        bbox = torch.tensor([0.0, 0.0, float(w), float(h)], dtype=torch.float32)
        target = {
            'species': torch.tensor(self.mappings['species'][s.species], dtype=torch.long),
            'genus': torch.tensor(self.mappings['genus'][s.genus], dtype=torch.long),
            'family': torch.tensor(self.mappings['family'][s.family], dtype=torch.long),
            'order': torch.tensor(self.mappings['order'][s.order], dtype=torch.long),
            'bbox': bbox,
            'objectness': torch.tensor(1.0, dtype=torch.float32),
        }
        return img, target


def _train_val_indices(samples: list[Sample], val_split: float, seed: int) -> tuple[list[int], list[int]]:
    n = len(samples)
    if n == 0:
        raise ValueError('No images found in discovered fishnet root. Verify Kaggle dataset mount and structure.')
    if n == 1:
        return [0], [0]

    val_split = float(val_split)
    val_split = min(max(val_split, 0.05), 0.5)

    idx = list(range(n))
    labels = [samples[i].species for i in idx]
    min_count = min(labels.count(x) for x in set(labels))
    stratify = labels if len(set(labels)) > 1 and min_count >= 2 else None

    try:
        tr, va = train_test_split(idx, test_size=val_split, random_state=seed, stratify=stratify)
    except ValueError:
        tr, va = train_test_split(idx, test_size=val_split, random_state=seed, stratify=None)

    if len(tr) == 0:
        tr = va[:1]
    if len(va) == 0:
        va = tr[:1]
    return tr, va


def build_datasets(cfg: dict, transforms: dict[str, Any]):
    root = discover_fishnet_root(cfg['data'].get('dataset_root'))
    samples = scan_samples(root)
    if cfg['data'].get('max_samples'):
        samples = samples[: int(cfg['data']['max_samples'])]

    species = sorted({s.species for s in samples})
    genus = sorted({s.genus for s in samples})
    family = sorted({s.family for s in samples})
    order = sorted({s.order for s in samples})
    mappings = {
        k: {v: i for i, v in enumerate(vals)}
        for k, vals in [('species', species), ('genus', genus), ('family', family), ('order', order)]
    }

    tr_idx, va_idx = _train_val_indices(samples, cfg['data'].get('val_split', 0.2), cfg.get('seed', 42))

    train_ds = FishNetDataset([samples[i] for i in tr_idx], mappings, cfg['data']['image_size'], transforms['train'])
    val_ds = FishNetDataset([samples[i] for i in va_idx], mappings, cfg['data']['image_size'], transforms['val'])

    class_freq: dict[str, int] = {}
    for s in samples:
        class_freq[s.species] = class_freq.get(s.species, 0) + 1

    meta = {
        'root': str(root),
        'num_samples': len(samples),
        'num_species': len(species),
        'num_genus': len(genus),
        'num_family': len(family),
        'num_order': len(order),
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'class_frequency': class_freq,
        'mappings': mappings,
    }

    if cfg['data'].get('cache_metadata', True):
        cache_path = Path(cfg.get('output_dir', 'runs/lhlfnetpp')) / 'dataset_cache.json'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    return train_ds, val_ds, meta
