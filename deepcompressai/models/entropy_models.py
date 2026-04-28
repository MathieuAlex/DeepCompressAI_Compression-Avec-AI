"""Differentiable entropy models.

During **training** both models add uniform noise to approximate quantisation
and return a per-element likelihood used for rate estimation.

During **inference** they perform hard rounding (quantisation) and return
the quantised tensor together with an estimated bit-count derived from the
learned probability model.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyBottleneck(nn.Module):
    """Factorised prior entropy model (Ballé et al. 2018).

    Models the marginal distribution of each channel of a latent tensor
    independently using a learned, non-parametric cumulative distribution.

    Parameters
    ----------
    channels:
        Number of channels in the latent tensor (``M`` or ``K``).
    init_scale:
        Initial scale of the density (controls variance at initialisation).
    num_filters:
        Number of hidden units in each density transform layer.
    """

    def __init__(
        self,
        channels: int,
        init_scale: float = 10.0,
        num_filters: int = 3,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.init_scale = init_scale
        filters = [1] + [num_filters] * 3 + [1]

        # Per-channel learnable parameters for the cumulative transform
        self._matrices: nn.ParameterList = nn.ParameterList()
        self._biases: nn.ParameterList = nn.ParameterList()
        self._factors: nn.ParameterList = nn.ParameterList()

        scale = init_scale ** (1.0 / (len(filters) - 1))
        for i in range(len(filters) - 1):
            in_f, out_f = filters[i], filters[i + 1]
            # Matrix: (channels, out_f, in_f)
            mat = torch.Tensor(channels, out_f, in_f)
            nn.init.orthogonal_(mat)
            self._matrices.append(nn.Parameter(mat * math.log(scale)))
            bias = torch.Tensor(channels, out_f, 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self._biases.append(nn.Parameter(bias))
            factor = torch.zeros(channels, out_f, 1)
            self._factors.append(nn.Parameter(factor))

    # ------------------------------------------------------------------
    # Cumulative distribution
    # ------------------------------------------------------------------

    def _logits_cumulative(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate log-odds of the CDF at ``inputs``.

        Parameters
        ----------
        inputs:
            Tensor of shape ``(channels, 1, N)`` where the last dimension is
            any number of sample points.
        """
        logits = inputs
        for mat, bias, factor in zip(self._matrices, self._biases, self._factors):
            # mat: (C, out, in), logits: (C, in, N)
            mat_pos = F.softplus(mat)
            logits = torch.bmm(mat_pos, logits) + bias
            logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise ``z`` and return the quantised tensor and per-element bits.

        During training uniform noise is added; during inference hard rounding
        is applied.

        Parameters
        ----------
        z:
            Latent tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        z_hat:
            Noise-perturbed (training) or rounded (eval) tensor, same shape as
            ``z``.
        bits:
            Scalar tensor – estimated total bits for the batch.
        """
        B, C, H, W = z.shape

        if self.training:
            noise = torch.empty_like(z).uniform_(-0.5, 0.5)
            z_hat = z + noise
        else:
            z_hat = z.round()

        # Likelihood: p(z_hat) ≈ CDF(z_hat + 0.5) - CDF(z_hat - 0.5)
        z_flat = z_hat.permute(1, 0, 2, 3).reshape(C, 1, -1)  # (C, 1, B*H*W)
        upper = self._logits_cumulative(z_flat + 0.5)
        lower = self._logits_cumulative(z_flat - 0.5)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        likelihood = likelihood.clamp(min=1e-9)

        bits = -likelihood.log2().sum()
        return z_hat, bits


class GaussianConditional(nn.Module):
    """Conditional Gaussian entropy model for the main latent ``y``.

    Given predicted Gaussian parameters from the hyperprior, this module
    quantises ``y`` and estimates the bit-rate.

    Parameters
    ----------
    scale_bound:
        Minimum allowed scale (standard deviation) to keep likelihoods finite.
    """

    def __init__(self, scale_bound: float = 0.11) -> None:
        super().__init__()
        self.scale_bound = scale_bound

    def _standardised_cumulative(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard-Normal CDF computed via the error function."""
        return 0.5 * (1.0 + torch.erf(inputs / math.sqrt(2)))

    def forward(
        self,
        y: torch.Tensor,
        scales: torch.Tensor,
        means: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise ``y`` and return the quantised tensor and per-element bits.

        Parameters
        ----------
        y:
            Main latent tensor of shape ``(B, M, H', W')``.
        scales:
            Predicted Gaussian standard deviations, same shape as ``y``.
        means:
            Optional predicted Gaussian means, same shape as ``y``.  When
            ``None`` the distribution is zero-mean.

        Returns
        -------
        y_hat:
            Noise-perturbed (training) or rounded (eval) tensor.
        bits:
            Scalar tensor – estimated total bits for the batch.
        """
        scales = scales.clamp(min=self.scale_bound)

        if self.training:
            noise = torch.empty_like(y).uniform_(-0.5, 0.5)
            y_hat = y + noise
        else:
            y_hat = (y - (means if means is not None else 0.0)).round()
            if means is not None:
                y_hat = y_hat + means

        # Likelihood: P(y_hat - 0.5 ≤ Y ≤ y_hat + 0.5 | N(mean, scale²))
        y_centred = y_hat - (means if means is not None else 0.0)
        upper = self._standardised_cumulative((y_centred + 0.5) / scales)
        lower = self._standardised_cumulative((y_centred - 0.5) / scales)
        likelihood = (upper - lower).clamp(min=1e-9)

        bits = -likelihood.log2().sum()
        return y_hat, bits
