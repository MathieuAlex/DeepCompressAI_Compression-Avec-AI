"""Generalised Divisive Normalisation (GDN) and its inverse (IGDN).

Reference
---------
Ballé et al., "Density Modeling of Images Using a Generalized Normalization
Transformation", ICLR 2016.  https://arxiv.org/abs/1511.06281
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    """Generalised Divisive Normalisation (GDN / IGDN).

    Parameters
    ----------
    num_channels:
        Number of feature-map channels ``C``.
    inverse:
        When ``True`` implements IGDN (the synthesis/decoder variant).
    beta_min:
        Small additive constant that keeps the denominator positive.
    gamma_init:
        Initial value on the diagonal of the ``gamma`` matrix.
    """

    def __init__(
        self,
        num_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min

        # β_i > 0 — one scalar per channel
        self.beta = nn.Parameter(torch.ones(num_channels))

        # γ_{ij} ≥ 0 — full C×C matrix initialised to a scaled identity
        self.gamma = nn.Parameter(gamma_init * torch.eye(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce non-negativity
        beta = F.softplus(self.beta) + self.beta_min  # (C,)
        gamma = F.relu(self.gamma)  # (C, C)

        # Compute the norm: n_i = sqrt( beta_i + sum_j gamma_{ij} * x_j^2 )
        x_sq = x.pow(2)  # (B, C, H, W)
        # gamma @ x_sq: einsum over the channel dimension
        norm_sq = (
            torch.einsum("ij,bjhw->bihw", gamma, x_sq)
            + beta.view(1, -1, 1, 1)
        )  # (B, C, H, W)
        norm = norm_sq.sqrt()

        if self.inverse:
            # IGDN: y = x * norm
            return x * norm
        # GDN: y = x / norm
        return x / norm
