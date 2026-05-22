# LHLF-Net++

Research-grade PyTorch implementation of **LHLF-Net++: Lightweight Hierarchical Long-Tail Unified Detection and Taxonomy Framework for Fine-Grained Underwater Fish Perception**.

## Features
- SegFormer/MiT lightweight hierarchical encoder (timm)
- Lightweight FPN with anatomical attention
- Unified one-stage outputs: objectness + bbox + taxonomy heads
- Multi-head taxonomy prediction: order/family/genus/species
- Confidence-aware taxonomy fallback inference
- Long-tail friendly unified taxonomy supervision
- Kaggle-first dynamic dataset discovery for random FishNet mount folder names

## Kaggle dataset discovery
The loader automatically discovers:
`/kaggle/input/datasets/<random_folder>/fishnet/`

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py --config configs/default.yaml
```

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

## FishNet assumptions handled
- recursive image scanning
- species discovery from folder names
- taxonomy parsing heuristics from species names
- corrupt image skipping with robust fallback image
- auto train/val split if predefined split absent

## Notes
This implementation supports unified classification/detection-style training from image folders. If explicit detection annotations are later added, loader extension points are available in `datasets/fishnet.py`.
