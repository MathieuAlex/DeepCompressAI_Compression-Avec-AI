"""Analysis transform (encoder): image → latent representation y.

Architecture follows the scale-hyperprior model of Ballé et al. 2018
(https://arxiv.org/abs/1802.01436).  Four strided convolution + GDN stages
progressively down-sample the spatial resolution by a factor of 16.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gdn import GDN


class AnalysisTransform(nn.Module):
    """Encoder that maps a 3-channel image to a latent tensor ``y``.

    Parameters
    ----------
    num_filters:
        Number of intermediate feature channels ``N``.
    num_latent_channels:
        Number of output (latent) channels ``M``.
    """

    def __init__(self, num_filters: int = 128, num_latent_channels: int = 192) -> None:
        super().__init__()
        N, M = num_filters, num_latent_channels

        self.transform = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input image tensor of shape ``(B, 3, H, W)`` with values in
            ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Latent tensor ``y`` of shape ``(B, M, H/16, W/16)``.
        """
        return self.transform(x)
