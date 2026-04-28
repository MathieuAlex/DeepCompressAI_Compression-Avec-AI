"""Evaluator: runs a full evaluation loop over a dataset.

Reports per-image PSNR, MS-SSIM, BPP, and compression ratio, then
aggregates them into summary statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepcompressai.models.compressor import ScaleHyperprior
from deepcompressai.evaluation.metrics import compute_psnr, compute_ms_ssim


class Evaluator:
    """Evaluate a trained :class:`ScaleHyperprior` on a test / validation set.

    Parameters
    ----------
    model:
        Trained compression model.
    device:
        ``'cuda'`` or ``'cpu'``.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        model: ScaleHyperprior,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Run the evaluation loop.

        Parameters
        ----------
        loader:
            DataLoader that yields image batches (``split='val'`` or
            ``split='test'``).

        Returns
        -------
        dict
            Keys: ``"psnr"``, ``"ms_ssim"``, ``"bpp"``, ``"compression_ratio"``.
            Values: mean over all images in the loader.
        """
        psnr_list: List[float] = []
        ms_ssim_list: List[float] = []
        bpp_list: List[float] = []

        for images in tqdm(loader, desc="Evaluating", leave=True):
            images = images.to(self.device, non_blocking=True)
            B, _, H, W = images.shape
            num_pixels = H * W

            compressed = self.model.compress(images)
            total_bits = (
                compressed["y_bits"] + compressed["z_bits"]
            ).item()
            bpp = total_bits / (B * num_pixels)

            x_hat = self.model.decompress(compressed["y_hat"])

            psnr = compute_psnr(x_hat, images).item()
            psnr_list.append(psnr)
            bpp_list.append(bpp)

            try:
                ms_ssim_val = compute_ms_ssim(x_hat, images).item()
                ms_ssim_list.append(ms_ssim_val)
            except ImportError:
                pass

        results: Dict[str, float] = {
            "psnr": sum(psnr_list) / len(psnr_list),
            "bpp": sum(bpp_list) / len(bpp_list),
        }

        # Original image uses 3 channels × 8 bits/pixel
        original_bpp = 24.0
        results["compression_ratio"] = original_bpp / results["bpp"] if results["bpp"] > 0 else 0.0

        if ms_ssim_list:
            results["ms_ssim"] = sum(ms_ssim_list) / len(ms_ssim_list)

        return results

    def report(self, loader: DataLoader) -> None:
        """Evaluate and print a formatted summary."""
        results = self.evaluate(loader)
        print("\n── Evaluation Results ──────────────────────")
        print(f"  PSNR              : {results['psnr']:.2f} dB")
        if "ms_ssim" in results:
            print(f"  MS-SSIM           : {results['ms_ssim']:.4f}")
        print(f"  BPP               : {results['bpp']:.4f}")
        print(f"  Compression ratio : {results['compression_ratio']:.1f}:1")
        print("────────────────────────────────────────────\n")
