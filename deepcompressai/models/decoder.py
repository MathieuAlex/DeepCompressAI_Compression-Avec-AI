"""Synthesis transform (decoder): quantised latent ŷ → reconstructed image x̂.

Architecture is the mirror image of the analysis transform: four strided
transposed-convolution + IGDN stages up-sample the spatial resolution back to
the original size.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gdn import GDN


class SynthesisTransform(nn.Module):
    """Decoder that maps the quantised latent ``y_hat`` back to image space.

    Parameters
    ----------
    num_filters:
        Number of intermediate feature channels ``N``.
    num_latent_channels:
        Number of input (latent) channels ``M``.
    """

    def __init__(self, num_filters: int = 128, num_latent_channels: int = 192) -> None:
        super().__init__()
        N, M = num_filters, num_latent_channels

        self.transform = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_hat:
            Quantised latent tensor of shape ``(B, M, H', W')``.

        Returns
        -------
        torch.Tensor
            Reconstructed image ``x_hat`` of shape ``(B, 3, 16·H', 16·W')``,
            clamped to ``[0, 1]``.
        """
        return self.transform(y_hat).clamp(0.0, 1.0)
