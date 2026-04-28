#!/usr/bin/env python3
"""Training entry point for the DeepCompressAI learned compression model.

Usage
-----
    python scripts/train.py \\
        --train-dir data/train \\
        --val-dir   data/val \\
        --epochs    200 \\
        --lmbda     0.01 \\
        --batch-size 16 \\
        --patch-size 256 \\
        --num-filters 128 \\
        --num-latent-channels 192 \\
        --lr 1e-4 \\
        --checkpoint-dir checkpoints/lmbda0.01 \\
        --wandb

Add ``--resume checkpoints/lmbda0.01/latest.pt`` to resume from a checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from deepcompressai.data import build_dataloader
from deepcompressai.models import ScaleHyperprior
from deepcompressai.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Scale Hyperprior image compression model."
    )
    # Data
    parser.add_argument("--train-dir", required=True, help="Training image directory.")
    parser.add_argument("--val-dir", required=True, help="Validation image directory.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-latent-channels", type=int, default=192)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lmbda", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--distortion", choices=["mse", "ms-ssim"], default="mse")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume.")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default="deepcompressai")

    # Hardware
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("── DeepCompressAI Training ─────────────────")
    print(f"  λ              : {args.lmbda}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Patch size     : {args.patch_size}")
    print(f"  Num filters    : {args.num_filters}")
    print(f"  Latent channels: {args.num_latent_channels}")
    print(f"  Device         : {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("────────────────────────────────────────────\n")

    train_loader = build_dataloader(
        args.train_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        split="train",
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        args.val_dir,
        batch_size=1,
        patch_size=None,
        split="val",
        num_workers=args.num_workers,
    )

    model = ScaleHyperprior(
        num_filters=args.num_filters,
        num_latent_channels=args.num_latent_channels,
    )
    print(f"Model parameters: {model.num_parameters():,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lmbda=args.lmbda,
        lr=args.lr,
        aux_lr=args.aux_lr,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()
