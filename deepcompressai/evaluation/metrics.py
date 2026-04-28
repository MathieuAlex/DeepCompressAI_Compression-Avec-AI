"""Image quality metrics for compression evaluation.

Provides:

* :func:`compute_psnr`    – Peak Signal-to-Noise Ratio (dB)
* :func:`compute_ms_ssim` – Multi-Scale Structural Similarity Index
"""

from __future__ import annotations

import math

import torch

try:
    from pytorch_msssim import ms_ssim as _ms_ssim
    _MSSSIM_AVAILABLE = True
except ImportError:
    _MSSSIM_AVAILABLE = False


def compute_psnr(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the mean PSNR (dB) over a batch of images.

    Parameters
    ----------
    x_hat:
        Reconstructed images, shape ``(B, C, H, W)``, values in ``[0, 1]``.
    x:
        Reference images, same shape and range.

    Returns
    -------
    torch.Tensor
        Scalar tensor with the mean PSNR in dB.
    """
    mse = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3))  # per-image MSE
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean()


def compute_ms_ssim(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the mean MS-SSIM over a batch of images.

    Requires the ``pytorch_msssim`` package.

    Parameters
    ----------
    x_hat:
        Reconstructed images, shape ``(B, C, H, W)``, values in ``[0, 1]``.
    x:
        Reference images, same shape and range.

    Returns
    -------
    torch.Tensor
        Scalar tensor with the mean MS-SSIM (higher is better).

    Raises
    ------
    ImportError
        When ``pytorch_msssim`` is not installed.
    """
    if not _MSSSIM_AVAILABLE:
        raise ImportError(
            "pytorch_msssim is required.  Install it with: pip install pytorch-msssim"
        )
    return _ms_ssim(x_hat, x, data_range=1.0, size_average=True)
