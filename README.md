# LHLF-Net++
Research-grade PyTorch implementation for underwater fish detection and taxonomy-aware multi-task learning.

## Installation
```bash
pip install -r requirements.txt
```

## Dataset layout
Uses `./dataset` exclusively.

```
./dataset/
  images/{train,val,test}
  annotations/{train.json,val.json,test.json,taxonomy.json,traits.json}
```

## Training
```bash
bash scripts/train.sh
```

## Evaluation
```bash
bash scripts/eval.sh
```

## Inference
```bash
bash scripts/infer.sh
```

## Architecture
Input → Hierarchical Transformer Encoder → Multi-Scale Fusion → FPN → Anatomical Attention → Detection + Shared Embedding + Taxonomy Heads.

## Output dictionary
- detections
- order_logits
- family_logits
- genus_logits
- species_logits
- trait_logits
