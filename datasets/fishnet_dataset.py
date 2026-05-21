from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetPaths:
    root: Path
    train_images: Path
    val_images: Path
    test_images: Path
    train_annotations: Path
    val_annotations: Path
    test_annotations: Path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_dataset_paths(cfg_dataset: Dict[str, str]) -> DatasetPaths:
    candidates = [
        Path("/kaggle/input/datasets/gbhjgghn"),
        Path("./dataset/fishnet").resolve(),
        Path(cfg_dataset.get("root", "./dataset")).resolve(),
    ]

    for root in candidates:
        if not root.exists():
            continue
        annotations = root / "annotations"
        images = root / "images"
        if annotations.exists() and images.exists():
            return DatasetPaths(
                root=root,
                train_images=images / "train",
                val_images=images / "val",
                test_images=images / "test",
                train_annotations=annotations / "train.json",
                val_annotations=annotations / "val.json",
                test_annotations=annotations / "test.json",
            )

    cfg_root = Path(cfg_dataset.get("root", "./dataset")).resolve()
    return DatasetPaths(
        root=cfg_root,
        train_images=Path(cfg_dataset.get("train_images", cfg_root / "images/train")),
        val_images=Path(cfg_dataset.get("val_images", cfg_root / "images/val")),
        test_images=Path(cfg_dataset.get("test_images", cfg_root / "images/test")),
        train_annotations=Path(cfg_dataset.get("train_annotations", cfg_root / "annotations/train.json")),
        val_annotations=Path(cfg_dataset.get("val_annotations", cfg_root / "annotations/val.json")),
        test_annotations=Path(cfg_dataset.get("test_annotations", cfg_root / "annotations/test.json")),
    )


def validate_dataset(paths: DatasetPaths) -> None:
    missing: List[str] = []
    required = [
        paths.root,
        paths.train_images,
        paths.val_images,
        paths.train_annotations,
        paths.val_annotations,
    ]
    for p in required:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Dataset validation failed. Missing paths:\n" + "\n".join(missing))

    tr = _load_json(paths.train_annotations)
    va = _load_json(paths.val_annotations)
    print("[Dataset] root:", paths.root)
    print("[Dataset] train images:", len(tr.get("images", [])), "annotations:", len(tr.get("annotations", [])))
    print("[Dataset] val images:", len(va.get("images", [])), "annotations:", len(va.get("annotations", [])))


class FishNetDataset(Dataset):
    def __init__(self, image_dir: str | Path, ann_path: str | Path, transforms=None):
        self.image_dir = Path(image_dir)
        self.coco = _load_json(Path(ann_path))
        self.transforms = transforms
        self.images = self.coco["images"]
        self.ann_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.coco["annotations"]:
            self.ann_by_img.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, file_name: str) -> np.ndarray:
        path = self.image_dir / file_name
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        im = self.images[idx]
        img = self._load_image(im["file_name"])
        anns = self.ann_by_img.get(im["id"], [])

        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann.get("category_id", 0)))

        if not boxes:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        first = anns[0] if anns else {}
        trait = np.array(first.get("trait_vector", [0] * 32), dtype=np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = [list(b) for b in transformed["bboxes"]] or [[0.0, 0.0, 1.0, 1.0]]
            labels = list(transformed["labels"]) or [0]

        return {
            "image": img,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "order_id": torch.tensor(int(first.get("order_id", 0)), dtype=torch.long),
            "family_id": torch.tensor(int(first.get("family_id", 0)), dtype=torch.long),
            "genus_id": torch.tensor(int(first.get("genus_id", 0)), dtype=torch.long),
            "species_id": torch.tensor(int(first.get("species_id", 0)), dtype=torch.long),
            "trait_vector": torch.tensor(trait, dtype=torch.float32),
        }
