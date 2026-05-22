# LHLF-Net++

PyTorch implementation of **LHLF-Net++: Lightweight Hierarchical Long-Tail Unified Detection and Taxonomy Framework for Fine-Grained Underwater Fish Perception** with Kaggle-first data discovery.

## What is implemented
- MiT/SegFormer-style lightweight hierarchical encoder (`timm` features-only)
- Lightweight FPN multi-scale fusion
- Anatomical attention (channel + spatial)
- Unified head for objectness, bbox regression, taxonomy logits, and shared embedding
- Multi-head taxonomy outputs: order/family/genus/species
- Confidence-aware taxonomy fallback inference
- AMP-enabled training/evaluation/inference

## Kaggle dataset discovery
Dataset root is auto-discovered by searching for a `fishnet` folder in:
- `/kaggle/input/datasets/`
- `/kaggle/input/`
- current working directory

Expected structure:

```
fishnet/
  species_A/
    *.jpg|*.png|...
  species_B/
    nested/folders/also/supported/*.jpg
```

If auto discovery fails, set explicit root in config:

```yaml
data:
  dataset_root: /kaggle/input/datasets/<random>/fishnet
```

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py --config configs/default.yaml
```

Artifacts are stored in `runs/lhlfnetpp/`:
- `best.ckpt`
- `last.ckpt`
- `dataset_meta.json`
- `dataset_cache.json`

## Evaluate
```bash
python evaluate.py --config configs/default.yaml --ckpt runs/lhlfnetpp/best.ckpt
```

## Inference
```bash
python infer.py --config configs/default.yaml --ckpt runs/lhlfnetpp/best.ckpt --image /path/to/image.jpg
```

## ONNX export
```bash
python export_onnx.py --config configs/default.yaml --species 500 --out lhlfnetpp.onnx
```

## Troubleshooting
### Error: `ValueError: With n_samples=0 ...`
No images were discovered. Verify:
1. Kaggle dataset is attached to the notebook.
2. There is a real `fishnet/` directory with species subfolders.
3. Images use supported extensions (`jpg/jpeg/png/bmp/webp/tif/tiff`).
4. If your mount is custom, set `data.dataset_root` explicitly.

### Very small dataset
If there are very few images/species, splitting gracefully falls back to non-stratified split and guarantees non-empty train/val partitions.
