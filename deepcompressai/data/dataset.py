"""Image dataset for training and evaluating learned compression models.

Supports arbitrary image directories. Images are converted to RGB, optionally
cropped/resized, and returned as ``[0, 1]``-normalised float tensors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageCompressionDataset(Dataset):
    """Dataset that loads all images found recursively under ``root``.

    Parameters
    ----------
    root:
        Directory to search for images.
    patch_size:
        If provided, randomly crop each image to ``(patch_size, patch_size)``
        during training.  Pass ``None`` to disable cropping (e.g. for
        evaluation, where full images are typically used).
    split:
        One of ``'train'``, ``'val'``, or ``'test'``.  Only used to select the
        appropriate transform; the caller is responsible for pointing ``root``
        at the right sub-directory.
    transform:
        Optional custom transform applied *after* the default one.  When
        provided it *replaces* the default transform.
    extensions:
        Image file extensions to look for (case-insensitive).
    """

    _EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")

    def __init__(
        self,
        root: Union[str, Path],
        patch_size: Optional[int] = 256,
        split: str = "train",
        transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.root}")

        exts = extensions or self._EXTENSIONS
        self.paths: List[Path] = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in exts
        )
        if not self.paths:
            raise RuntimeError(f"No images found under '{self.root}'.")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_transform(split, patch_size)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transform(split: str, patch_size: Optional[int]) -> Callable:
        ops: List[Callable] = []
        if split == "train":
            if patch_size is not None:
                ops.append(transforms.RandomCrop(patch_size))
            ops.append(transforms.RandomHorizontalFlip())
        else:
            # Pad to a multiple of 64 so that strided conv/deconv layers
            # produce shapes that divide evenly during inference.
            ops.append(transforms.Lambda(lambda img: _pad_to_multiple(img, 64)))

        ops.append(transforms.ToTensor())
        return transforms.Compose(ops)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pad_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
    """Pad ``img`` on the right/bottom so its dimensions are multiples of ``multiple``."""
    w, h = img.size
    pad_w = (multiple - w % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    if pad_w == 0 and pad_h == 0:
        return img
    padded = Image.new(img.mode, (w + pad_w, h + pad_h), 0)
    padded.paste(img, (0, 0))
    return padded


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataloader(
    root: Union[str, Path],
    batch_size: int,
    patch_size: Optional[int] = 256,
    split: str = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Return a :class:`~torch.utils.data.DataLoader` for ``root``.

    Parameters
    ----------
    root:
        Path to the image directory (e.g. ``data/train``).
    batch_size:
        Number of images per batch.
    patch_size:
        Random-crop patch size for training; ``None`` disables cropping.
    split:
        ``'train'``, ``'val'``, or ``'test'``.
    num_workers:
        Number of data-loading worker processes.
    pin_memory:
        Pin tensors to CUDA-accessible memory for faster GPU transfers.
    """
    dataset = ImageCompressionDataset(root, patch_size=patch_size, split=split)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
