"""Scale Hyperprior – full learned image compression model.

Implements the variational model from:

    Ballé et al., "Variational Image Compression with a Scale Hyperprior",
    ICLR 2018.  https://arxiv.org/abs/1802.01436

The model consists of:

* :class:`AnalysisTransform`         – image → latent y
* :class:`HyperAnalysisTransform`    – |y| → side info z
* :class:`EntropyBottleneck`         – entropy model for z
* :class:`HyperSynthesisTransform`   – ẑ → Gaussian scales for y
* :class:`GaussianConditional`       – entropy model for y | ẑ
* :class:`SynthesisTransform`        – ŷ → reconstructed image x̂
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .encoder import AnalysisTransform
from .decoder import SynthesisTransform
from .hyperprior import HyperAnalysisTransform, HyperSynthesisTransform
from .entropy_models import EntropyBottleneck, GaussianConditional


class ScaleHyperprior(nn.Module):
    """Scale Hyperprior learned image compression model.

    Parameters
    ----------
    num_filters:
        Number of intermediate feature channels ``N``.
    num_latent_channels:
        Number of latent channels ``M`` (output of the analysis transform).
    num_hyper_filters:
        Number of channels in the side-information latent ``z``.  Defaults to
        ``num_filters``.
    """

    def __init__(
        self,
        num_filters: int = 128,
        num_latent_channels: int = 192,
        num_hyper_filters: int | None = None,
    ) -> None:
        super().__init__()
        K = num_hyper_filters if num_hyper_filters is not None else num_filters

        self.analysis = AnalysisTransform(num_filters, num_latent_channels)
        self.synthesis = SynthesisTransform(num_filters, num_latent_channels)
        self.hyper_analysis = HyperAnalysisTransform(
            num_filters, num_latent_channels, K
        )
        self.hyper_synthesis = HyperSynthesisTransform(
            num_filters, num_latent_channels, K, predict_mean=False
        )
        self.entropy_bottleneck = EntropyBottleneck(K)
        self.gaussian_conditional = GaussianConditional()

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full encode → quantise → decode pass.

        Parameters
        ----------
        x:
            Input image batch of shape ``(B, 3, H, W)`` with values in
            ``[0, 1]``.

        Returns
        -------
        dict with keys:

        * ``"x_hat"``  – reconstructed images, shape ``(B, 3, H, W)``
        * ``"y_bits"`` – scalar, estimated bits for ``y`` in the batch
        * ``"z_bits"`` – scalar, estimated bits for ``z`` in the batch
        """
        y = self.analysis(x)
        z = self.hyper_analysis(y)

        z_hat, z_bits = self.entropy_bottleneck(z)
        scales = self.hyper_synthesis(z_hat)

        y_hat, y_bits = self.gaussian_conditional(y, scales)
        x_hat = self.synthesis(y_hat)

        return {"x_hat": x_hat, "y_bits": y_bits, "z_bits": z_bits}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress an image and return the quantised latents and bit estimates.

        This is a *simulation* of compression: latents are rounded (quantised)
        and the theoretical rate is computed from the learned entropy model.
        Production deployments would replace the rate estimate with actual
        range-coded bitstreams.

        Parameters
        ----------
        x:
            Single image tensor of shape ``(1, 3, H, W)``.

        Returns
        -------
        dict with keys:

        * ``"y_hat"``  – quantised main latent, shape ``(1, M, H', W')``
        * ``"z_hat"``  – quantised side-info latent, shape ``(1, K, H'', W'')``
        * ``"y_bits"`` – estimated bits for ``y`` (scalar tensor)
        * ``"z_bits"`` – estimated bits for ``z`` (scalar tensor)
        """
        self.eval()
        y = self.analysis(x)
        z = self.hyper_analysis(y)

        z_hat, z_bits = self.entropy_bottleneck(z)
        scales = self.hyper_synthesis(z_hat)
        y_hat, y_bits = self.gaussian_conditional(y, scales)

        return {"y_hat": y_hat, "z_hat": z_hat, "y_bits": y_bits, "z_bits": z_bits}

    @torch.no_grad()
    def decompress(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image from a quantised latent.

        Parameters
        ----------
        y_hat:
            Quantised latent of shape ``(1, M, H', W')``.

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape ``(1, 3, H, W)``, values in
            ``[0, 1]``.
        """
        self.eval()
        return self.synthesis(y_hat)

    # ------------------------------------------------------------------
    # Convenience: number of parameters
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
