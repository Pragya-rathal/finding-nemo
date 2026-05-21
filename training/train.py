from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.collate import fish_collate
from datasets.fishnet_dataset import FishNetDataset, discover_dataset_paths, validate_dataset
from datasets.transforms import build_transforms
from models.lhlfnetpp import LHLFNetPP
from models.losses import UnifiedLoss
from training.amp_utils import get_scaler
from training.engine import train_one_epoch
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.validate import validate
from utils.checkpoint import save_checkpoint
from utils.device import get_device
from utils.pathing import ensure_dir


def run_training(cfg):
    device = get_device()
    dataset_paths = discover_dataset_paths(cfg["dataset"])
    validate_dataset(dataset_paths)

    train_set = FishNetDataset(dataset_paths.train_images, dataset_paths.train_annotations, build_transforms(cfg["model"]["image_size"], True))
    val_set = FishNetDataset(dataset_paths.val_images, dataset_paths.val_annotations, build_transforms(cfg["model"]["image_size"], False))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=cfg["system"]["pin_memory"],
        collate_fn=fish_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=cfg["system"]["pin_memory"],
        collate_fn=fish_collate,
    )

    model = LHLFNetPP(cfg).to(device)
    loss_fn = UnifiedLoss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = get_scaler(cfg["train"]["amp"] and torch.cuda.is_available())

    out_dir = ensure_dir(Path(cfg["output_dir"]).resolve())
    writer = SummaryWriter(str(out_dir / "tb"))

    best = float("inf")
    patience = 0
    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            scaler,
            device,
            epoch,
            cfg["train"]["accumulation_steps"],
            writer,
        )
        val_metrics = validate(model, val_loader, loss_fn, device)
        writer.add_scalar("val/loss", val_metrics["val/loss"], epoch)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_metrics['val/loss']:.4f}")

        if val_metrics["val/loss"] < best:
            best = val_metrics["val/loss"]
            patience = 0
            save_checkpoint(str(out_dir / "best.pt"), {"model": model.state_dict(), "epoch": epoch, "best": best, "cfg": cfg})
        else:
            patience += 1

        if patience >= cfg["train"]["early_stopping"]["patience"]:
            print("[Training] Early stopping triggered.")
            break

    writer.close()
