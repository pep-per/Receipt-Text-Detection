#!/usr/bin/env python3
"""Pretrain the DBNet ResNet18 encoder with image-only MoCo v2."""

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule
from PIL import Image, ImageFile
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_ROOTS = {
    "official_train": PROJECT_ROOT / "data/datasets/images/train",
    "official_val": PROJECT_ROOT / "data/datasets/images/val",
    "official_test": PROJECT_ROOT / "data/datasets/images/test",
    "cord_v2": PROJECT_ROOT / "data/pseudo_label/cord-v2/images",
    "sroie": PROJECT_ROOT / "data/pseudo_label/sroie/images",
    "wildreceipt": PROJECT_ROOT / "data/pseudo_label/wildreceipt/images",
}
SOURCE_FAMILIES = {
    "official_train": "official",
    "official_val": "official",
    "official_test": "official",
    "cord_v2": "cord_v2",
    "sroie": "sroie",
    "wildreceipt": "wildreceipt",
}

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "baseline_code/outputs/v10_ssl_moco",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--encoder-momentum", type=float, default=0.996)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--memory-bank-size", type=int, default=4096)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--run-name", default="v10_ssl_moco_pretrain")
    parser.add_argument("--dry-run-steps", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_images():
    records = []
    for source, root in SOURCE_ROOTS.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing SSL image root: {root}")
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in paths:
            records.append(
                {
                    "path": str(path.resolve()),
                    "source": source,
                    "source_family": SOURCE_FAMILIES[source],
                }
            )
    if not records:
        raise RuntimeError("No SSL images found")
    return records


def write_manifest(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "source", "source_family"))
        writer.writeheader()
        writer.writerows(records)


class ReceiptImageDataset(Dataset):
    """Image-only dataset: annotation files are never accepted or opened."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            query, key = self.transform(image)
        return query, key, index


class MoCoModel(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, num_classes=0, global_pool="avg"
        )
        self.projection_head = MoCoProjectionHead(512, 2048, projection_dim)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, query_image, key_image, momentum):
        update_momentum(self.backbone, self.backbone_momentum, momentum)
        update_momentum(
            self.projection_head, self.projection_head_momentum, momentum
        )
        query = self.projection_head(self.backbone(query_image))
        with torch.no_grad():
            key = self.projection_head_momentum(self.backbone_momentum(key_image))
        return query, key


def make_loader(records, args):
    transform = MoCoV2Transform(
        input_size=args.input_size,
        cj_prob=0.5,
        cj_strength=0.2,
        min_scale=0.7,
        random_gray_scale=0.1,
        gaussian_blur=0.2,
        hf_prob=0.0,
        vf_prob=0.0,
        rr_prob=0.0,
    )
    dataset = ReceiptImageDataset(records, transform)
    family_counts = Counter(record["source_family"] for record in records)
    weights = [1.0 / family_counts[record["source_family"]] for record in records]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    return loader, family_counts


def make_lr_scheduler(optimizer, total_steps):
    def lr_scale(step):
        progress = min(step, total_steps) / max(total_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


def init_wandb(args, config):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project="receipt-text-detection",
        name=args.run_name,
        job_type="ssl-pretraining",
        mode=args.wandb_mode,
        dir=str(PROJECT_ROOT / "baseline_code"),
        config=config,
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_images()
    write_manifest(records, args.output_dir / "image_manifest.csv")
    loader, family_counts = make_loader(records, args)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    if args.dry_run_steps:
        total_steps = min(total_steps, args.dry_run_steps)

    source_counts = Counter(record["source"] for record in records)
    config = {
        "experiment": "V10 Domain Self-supervised Pilot",
        "algorithm": "MoCo v2",
        "library": "lightly==1.5.25",
        "label_access": "image pixels only; no annotation files opened",
        "evaluation_scope": "transductive local",
        "encoder": "timm resnet18 ImageNet initialization",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "learning_rate": args.learning_rate,
        "optimizer": "SGD",
        "lr_schedule": "step-wise cosine decay",
        "weight_decay": args.weight_decay,
        "optimizer_momentum": args.momentum,
        "encoder_momentum_start": args.encoder_momentum,
        "encoder_momentum_end": 1.0,
        "temperature": args.temperature,
        "memory_bank_size": args.memory_bank_size,
        "projection_dim": args.projection_dim,
        "source_sampling": "equal expected sampling across 4 source families",
        "source_counts": dict(source_counts),
        "source_family_counts": dict(family_counts),
        "images": len(records),
        "steps_per_epoch": steps_per_epoch,
        "seed": args.seed,
        "augmentation": {
            "min_crop_scale": 0.7,
            "color_jitter_probability": 0.5,
            "color_jitter_strength": 0.2,
            "grayscale_probability": 0.1,
            "gaussian_blur_probability": 0.2,
            "horizontal_flip_probability": 0.0,
            "vertical_flip_probability": 0.0,
            "random_rotation_probability": 0.0,
        },
    }
    with (args.output_dir / "pretrain_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    run = init_wandb(args, config)
    device = torch.device(args.device)
    model = MoCoModel(args.projection_dim).to(device)
    criterion = NTXentLoss(
        temperature=args.temperature,
        memory_bank_size=(args.memory_bank_size, args.projection_dim),
    ).to(device)
    optimizer = torch.optim.SGD(
        list(model.backbone.parameters()) + list(model.projection_head.parameters()),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = make_lr_scheduler(optimizer, args.epochs * steps_per_epoch)
    scaler = GradScaler(enabled=device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    history = []
    global_step = 0
    stopped = False
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        progress = tqdm(loader, desc=f"V10 SSL epoch {epoch + 1}/{args.epochs}")
        for query, key, _ in progress:
            query = query.to(device, non_blocking=True)
            key = key.to(device, non_blocking=True)
            encoder_momentum = cosine_schedule(
                global_step,
                args.epochs * steps_per_epoch,
                args.encoder_momentum,
                1.0,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                query_features, key_features = model(query, key, encoder_momentum)
                loss = criterion(query_features, key_features)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_samples = query.shape[0]
            epoch_loss += float(loss.detach()) * batch_samples
            epoch_samples += batch_samples
            global_step += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if run is not None:
                run.log(
                    {
                        "ssl/step_loss": float(loss.detach()),
                        "ssl/lr": optimizer.param_groups[0]["lr"],
                        "ssl/encoder_momentum": encoder_momentum,
                        "trainer/global_step": global_step,
                    },
                    step=global_step,
                )
            if args.dry_run_steps and global_step >= args.dry_run_steps:
                stopped = True
                break

        mean_loss = epoch_loss / max(epoch_samples, 1)
        epoch_record = {
            "epoch": epoch + 1,
            "loss": mean_loss,
            "samples": epoch_samples,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        if run is not None:
            run.log(
                {
                    "ssl/epoch": epoch + 1,
                    "ssl/epoch_loss": mean_loss,
                    "ssl/epoch_samples": epoch_samples,
                },
                step=global_step,
            )
        if stopped:
            break

    runtime_seconds = time.monotonic() - start
    checkpoint = {
        "encoder_state_dict": model.backbone.state_dict(),
        "projection_head_state_dict": model.projection_head.state_dict(),
        "encoder_momentum_state_dict": model.backbone_momentum.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "history": history,
        "global_step": global_step,
    }
    checkpoint_path = args.output_dir / "moco_final.pt"
    encoder_path = args.output_dir / "encoder_state_dict.pt"
    torch.save(checkpoint, checkpoint_path)
    torch.save(model.backbone.state_dict(), encoder_path)

    result = {
        "runtime_seconds": runtime_seconds,
        "global_steps": global_step,
        "final_epoch_loss": history[-1]["loss"],
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "encoder_checkpoint": str(encoder_path.resolve()),
        "dry_run": bool(args.dry_run_steps),
        "history": history,
    }
    with (args.output_dir / "pretrain_result.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    if run is not None:
        run.summary.update(result)
        run.finish()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
