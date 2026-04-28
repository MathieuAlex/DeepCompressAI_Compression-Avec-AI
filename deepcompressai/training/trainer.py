"""Training loop for the ScaleHyperprior model.

Features
--------
* Mixed-precision training (``torch.amp``)
* Gradient clipping
* Periodic validation
* ``wandb`` experiment tracking
* Checkpoint saving (best and most-recent)
* ``tqdm`` progress bars
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from deepcompressai.models.compressor import ScaleHyperprior
from deepcompressai.training.losses import RateDistortionLoss
from deepcompressai.evaluation.metrics import compute_psnr


class Trainer:
    """Orchestrates training and validation of a :class:`ScaleHyperprior` model.

    Parameters
    ----------
    model:
        The compression model to train.
    train_loader:
        DataLoader for training images.
    val_loader:
        DataLoader for validation images.
    lmbda:
        Rate-distortion trade-off λ.
    lr:
        Initial learning rate for Adam.
    aux_lr:
        Learning rate for auxiliary parameters of the entropy bottleneck.
    max_grad_norm:
        Maximum gradient norm for clipping (0 = disabled).
    checkpoint_dir:
        Directory where checkpoints are saved.
    use_wandb:
        Enable Weights & Biases logging.
    wandb_project:
        W&B project name.
    device:
        ``'cuda'`` or ``'cpu'``.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        model: ScaleHyperprior,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lmbda: float = 0.01,
        lr: float = 1e-4,
        aux_lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "deepcompressai",
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = RateDistortionLoss(lmbda=lmbda)

        # Two parameter groups: main network and auxiliary (entropy model)
        aux_params = {
            "params": list(model.entropy_bottleneck.parameters()),
            "lr": aux_lr,
        }
        main_params = {
            "params": [
                p
                for n, p in model.named_parameters()
                if "entropy_bottleneck" not in n
            ],
            "lr": lr,
        }
        self.optimizer = Adam([main_params, aux_params])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        self.use_wandb = use_wandb and _WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=wandb_project)
            wandb.watch(self.model, log_freq=100)

        self.best_val_loss = float("inf")
        self.epoch = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> dict:
        self.model.train()
        total_loss = total_rate = total_dist = 0.0
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f"Train [{self.epoch}]", leave=False) as bar:
            for images in bar:
                images = images.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda",
                ):
                    output = self.model(images)
                    losses = self.criterion(output, images)

                self.scaler.scale(losses["loss"]).backward()

                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += losses["loss"].item()
                total_rate += losses["rate"].item()
                total_dist += losses["distortion"].item()
                bar.set_postfix(
                    loss=f"{losses['loss'].item():.4f}",
                    bpp=f"{losses['rate'].item():.3f}",
                )

        return {
            "train/loss": total_loss / num_batches,
            "train/rate": total_rate / num_batches,
            "train/distortion": total_dist / num_batches,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = total_rate = total_dist = total_psnr = 0.0
        num_batches = len(self.val_loader)

        with tqdm(self.val_loader, desc=f"Val   [{self.epoch}]", leave=False) as bar:
            for images in bar:
                images = images.to(self.device, non_blocking=True)
                output = self.model(images)
                losses = self.criterion(output, images)

                total_loss += losses["loss"].item()
                total_rate += losses["rate"].item()
                total_dist += losses["distortion"].item()
                total_psnr += compute_psnr(output["x_hat"], images).item()
                bar.set_postfix(loss=f"{losses['loss'].item():.4f}")

        return {
            "val/loss": total_loss / num_batches,
            "val/rate": total_rate / num_batches,
            "val/distortion": total_dist / num_batches,
            "val/psnr": total_psnr / num_batches,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int) -> None:
        """Run the training loop for ``num_epochs`` epochs.

        Parameters
        ----------
        num_epochs:
            Total number of epochs to train.
        """
        for _ in range(num_epochs):
            self.epoch += 1
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            self.scheduler.step(val_metrics["val/loss"])

            if self.use_wandb:
                wandb.log({**train_metrics, **val_metrics, "epoch": self.epoch})

            print(
                f"Epoch {self.epoch:4d} | "
                f"loss {val_metrics['val/loss']:.4f} | "
                f"bpp {val_metrics['val/rate']:.3f} | "
                f"PSNR {val_metrics['val/psnr']:.2f} dB"
            )

            # Save latest checkpoint
            self._save_checkpoint("latest.pt")

            # Save best checkpoint
            if val_metrics["val/loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val/loss"]
                self._save_checkpoint("best.pt")

        if self.use_wandb:
            wandb.finish()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint.

        Parameters
        ----------
        path:
            Path to a ``.pt`` checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {self.epoch} (best val loss: {self.best_val_loss:.4f})")
