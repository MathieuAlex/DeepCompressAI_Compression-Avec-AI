#!/usr/bin/env python3
"""Evaluation entry point for the DeepCompressAI learned compression model.

Usage
-----
    python scripts/evaluate.py \\
        --test-dir  data/test \\
        --checkpoint checkpoints/lmbda0.01/best.pt \\
        --num-filters 128 \\
        --num-latent-channels 192

Prints PSNR (dB), MS-SSIM, average BPP, and compression ratio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from deepcompressai.data import build_dataloader
from deepcompressai.models import ScaleHyperprior
from deepcompressai.evaluation import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Scale Hyperprior model."
    )
    parser.add_argument("--test-dir", required=True, help="Test image directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-latent-channels", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = ScaleHyperprior(
        num_filters=args.num_filters,
        num_latent_channels=args.num_latent_channels,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    test_loader = build_dataloader(
        args.test_dir,
        batch_size=1,
        patch_size=None,
        split="test",
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    evaluator = Evaluator(model, device=str(device))
    evaluator.report(test_loader)


if __name__ == "__main__":
    main()
