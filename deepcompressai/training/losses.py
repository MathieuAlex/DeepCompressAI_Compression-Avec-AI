"""Rate-distortion loss for learned image compression.

The total loss is:

    L = λ · D(x, x̂)  +  R(y)  +  R(z)

where D is the distortion (MSE or MS-SSIM) and R(·) are the bit-rate
estimates returned by the entropy models.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ms_ssim
    _MSSSIM_AVAILABLE = True
except ImportError:
    _MSSSIM_AVAILABLE = False


class RateDistortionLoss(nn.Module):
    """Lagrangian rate-distortion loss.

    Parameters
    ----------
    lmbda:
        Trade-off parameter λ.  Higher values push towards lower distortion
        (higher quality) at the cost of more bits.  Typical values:
        ``[0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.05]``.
    distortion_metric:
        Either ``'mse'`` or ``'ms-ssim'``.
    """

    def __init__(
        self, lmbda: float = 0.01, distortion_metric: str = "mse"
    ) -> None:
        super().__init__()
        if distortion_metric not in ("mse", "ms-ssim"):
            raise ValueError(
                f"distortion_metric must be 'mse' or 'ms-ssim', got '{distortion_metric}'"
            )
        if distortion_metric == "ms-ssim" and not _MSSSIM_AVAILABLE:
            raise ImportError(
                "pytorch_msssim is required for ms-ssim loss.  "
                "Install it with: pip install pytorch-msssim"
            )
        self.lmbda = lmbda
        self.distortion_metric = distortion_metric

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the total loss and its components.

        Parameters
        ----------
        output:
            Dictionary returned by :meth:`ScaleHyperprior.forward` with keys
            ``"x_hat"``, ``"y_bits"``, and ``"z_bits"``.
        target:
            Original images, shape ``(B, 3, H, W)``.

        Returns
        -------
        dict with keys ``"loss"``, ``"distortion"``, ``"rate"``.
        """
        B, _, H, W = target.shape
        num_pixels = B * H * W

        # Rate (bits per pixel for the whole batch)
        rate = (output["y_bits"] + output["z_bits"]) / num_pixels

        # Distortion
        if self.distortion_metric == "mse":
            distortion = F.mse_loss(output["x_hat"], target)
        else:
            distortion = 1.0 - ms_ssim(
                output["x_hat"], target, data_range=1.0, size_average=True
            )

        loss = self.lmbda * 255.0**2 * distortion + rate

        return {"loss": loss, "distortion": distortion, "rate": rate}
