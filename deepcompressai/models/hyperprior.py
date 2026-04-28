"""Hyperprior transforms (hyper-encoder and hyper-decoder).

These modules implement the side-information path of the scale-hyperprior
model (Ballé et al. 2018):

* :class:`HyperAnalysisTransform`  encodes the latent ``y`` → side info ``z``
* :class:`HyperSynthesisTransform` decodes ``ẑ`` → Gaussian scale parameters
  for each element of ``y``
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HyperAnalysisTransform(nn.Module):
    """Hyper-encoder: maps the absolute-value of the latent to side information.

    Parameters
    ----------
    num_filters:
        Number of intermediate channels ``N``.
    num_latent_channels:
        Number of channels in ``y`` (input), ``M``.
    num_hyper_filters:
        Number of output channels for ``z``.  Defaults to ``N``.
    """

    def __init__(
        self,
        num_filters: int = 128,
        num_latent_channels: int = 192,
        num_hyper_filters: int | None = None,
    ) -> None:
        super().__init__()
        N = num_filters
        M = num_latent_channels
        K = num_hyper_filters if num_hyper_filters is not None else N

        self.transform = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, K, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y:
            Latent tensor of shape ``(B, M, H', W')``.

        Returns
        -------
        torch.Tensor
            Side-information tensor ``z`` of shape ``(B, K, H'', W'')``.
        """
        return self.transform(y.abs())


class HyperSynthesisTransform(nn.Module):
    """Hyper-decoder: maps the quantised side information to Gaussian parameters.

    Outputs ``(mu, sigma)`` for each element of the main latent ``y``.
    In the *scale* hyperprior variant (no mean prediction) only ``sigma`` is
    returned; set ``predict_mean=False`` (default) for that behaviour.

    Parameters
    ----------
    num_filters:
        Number of intermediate channels ``N``.
    num_latent_channels:
        Number of channels in ``y`` (output target), ``M``.
    num_hyper_filters:
        Number of input channels (``z`` channels).  Defaults to ``N``.
    predict_mean:
        If ``True``, predict both mean and scale (``2 * M`` output channels).
        If ``False``, predict scale only (``M`` output channels).
    """

    def __init__(
        self,
        num_filters: int = 128,
        num_latent_channels: int = 192,
        num_hyper_filters: int | None = None,
        predict_mean: bool = False,
    ) -> None:
        super().__init__()
        N = num_filters
        M = num_latent_channels
        K = num_hyper_filters if num_hyper_filters is not None else N
        out_channels = 2 * M if predict_mean else M

        self.predict_mean = predict_mean
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(K, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(N, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_hat:
            Quantised side-information tensor of shape ``(B, K, H'', W'')``.

        Returns
        -------
        torch.Tensor
            Gaussian parameter tensor of shape
            ``(B, 2*M, H', W')`` (mean + scale) when ``predict_mean=True``,
            or ``(B, M, H', W')`` (scale only) when ``predict_mean=False``.
            All scale values are guaranteed positive.
        """
        out = self.transform(z_hat)
        if self.predict_mean:
            M = out.shape[1] // 2
            mu, sigma = out[:, :M], out[:, M:]
            return torch.cat([mu, sigma.abs() + 1e-6], dim=1)
        return out.abs() + 1e-6
