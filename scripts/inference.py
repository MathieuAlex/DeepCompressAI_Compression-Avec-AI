#!/usr/bin/env python3
"""Inference script for the DeepCompressAI learned image compression model.

This script:

1. Loads a pre-trained :class:`~deepcompressai.models.ScaleHyperprior` from a
   checkpoint file.
2. Loads an input image and pre-processes it for the model.
3. **Compresses** the image through the analysis transform and entropy models.
4. **Decompresses** the quantised latent back to pixel space (``x_hat``).
5. **Saves** ``x_hat`` to disk as a PNG file.
6. **Prints** the bits-per-pixel (BPP) and compression ratio.

Usage
-----
    python scripts/inference.py \\
        --input  path/to/image.png \\
        --output path/to/xhat.png \\
        --checkpoint checkpoints/lmbda0.01/best.pt \\
        --num-filters 128 \\
        --num-latent-channels 192

Optional flags
--------------
    --device  cuda | cpu        (auto-detected)
    --show-stats                 print extra statistics (latent shape, etc.)

Notes
-----
Compression here is *simulated*: latents are hard-rounded (quantised) and the
theoretical rate is estimated from the learned entropy model rather than
produced as a real bitstream.  The workflow is:

    x ─► analysis ─► quantise ─► estimated_bpp
                         │
                    hyper-path
                         │
                         ▼
    x̂ ◄─ synthesis ◄── ŷ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from deepcompressai.models import ScaleHyperprior


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress an image with a pre-trained ScaleHyperprior model."
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="PATH",
        help="Path to the input image (PNG / JPEG / …).",
    )
    parser.add_argument(
        "--output", "-o", default="xhat.png", metavar="PATH",
        help="Where to save the decompressed output image (default: xhat.png).",
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True, metavar="PATH",
        help="Path to the .pt checkpoint produced by scripts/train.py.",
    )
    parser.add_argument(
        "--num-filters", type=int, default=128,
        help="Must match the value used during training (default: 128).",
    )
    parser.add_argument(
        "--num-latent-channels", type=int, default=192,
        help="Must match the value used during training (default: 192).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force 'cuda' or 'cpu'.  Auto-detected when omitted.",
    )
    parser.add_argument(
        "--show-stats", action="store_true",
        help="Print additional statistics (latent shapes, bit allocation, …).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 64) -> torch.Tensor:
    """Pad ``tensor`` on the right/bottom to the next multiple of ``multiple``."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))


def load_image(path: str) -> torch.Tensor:
    """Load an image and convert to a ``(1, 3, H, W)`` float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0)  # (1, 3, H, W)


def save_image(tensor: torch.Tensor, path: str, original_size: tuple[int, int]) -> None:
    """Save a ``(1, 3, H, W)`` tensor as a PNG, cropped to ``original_size``."""
    h, w = original_size
    cropped = tensor[:, :, :h, :w].squeeze(0)  # (3, H, W)
    img = TF.to_pil_image(cropped.clamp(0.0, 1.0))
    img.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = ScaleHyperprior(
        num_filters=args.num_filters,
        num_latent_channels=args.num_latent_channels,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Load image ──────────────────────────────────────────────────────────
    x = load_image(args.input).to(device)
    original_h, original_w = x.shape[2], x.shape[3]
    x_padded = _pad_to_multiple(x, multiple=64)

    _, _, H, W = x_padded.shape
    num_pixels = original_h * original_w

    print(f"Input image     : {args.input}  ({original_w}×{original_h} px)")

    # ── Compress ────────────────────────────────────────────────────────────
    print("Compressing …")
    compressed = model.compress(x_padded)

    y_hat = compressed["y_hat"]
    total_bits = (compressed["y_bits"] + compressed["z_bits"]).item()
    bpp = total_bits / num_pixels

    # ── Decompress ──────────────────────────────────────────────────────────
    print("Decompressing …")
    x_hat = model.decompress(y_hat)

    # ── Save output ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(x_hat, str(output_path), (original_h, original_w))
    print(f"Saved x_hat to  : {output_path}")

    # ── Statistics ──────────────────────────────────────────────────────────
    # Compression ratio relative to a raw RGB image (3 channels × 8 bits/px)
    raw_bpp = 24.0
    compression_ratio = raw_bpp / bpp if bpp > 0 else float("inf")

    print("\n── Compression Statistics ──────────────────")
    print(f"  Original size   : {original_w} × {original_h} px")
    print(f"  Raw BPP         : {raw_bpp:.1f}")
    print(f"  Compressed BPP  : {bpp:.4f}")
    print(f"  Compression ratio: {compression_ratio:.1f}:1")

    if args.show_stats:
        y_bits = compressed["y_bits"].item()
        z_bits = compressed["z_bits"].item()
        print(f"\n  Latent y shape  : {tuple(y_hat.shape)}")
        print(f"  Latent z shape  : {tuple(compressed['z_hat'].shape)}")
        print(f"  Bits (y)        : {y_bits:.0f}")
        print(f"  Bits (z)        : {z_bits:.0f}")
        print(f"  Total bits      : {total_bits:.0f}")

    print("────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
