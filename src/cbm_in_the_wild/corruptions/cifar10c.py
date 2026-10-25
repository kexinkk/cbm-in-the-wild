"""CIFAR-10-C loader (Hendrycks's distributed `.npy` files).

Distributed layout under `data/raw/CIFAR-10-C/`:
    <corruption>.npy   # uint8, shape (50 000, 32, 32, 3) — 5 severities × 10 000 imgs
    labels.npy         # int,   shape (50 000,) — labels tile per severity

Severity i ∈ {1..5} occupies indices [(i-1)*10_000 : i*10_000] in both arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cbm_in_the_wild.paths import cifar10_c_dir

# Natural-degradation types selected for evaluation.
SELECTED_CORRUPTIONS = ("motion_blur", "gaussian_noise", "jpeg_compression", "fog")

_PER_SEVERITY = 10_000


def load_cifar10c(
    corruption: str,
    severity: int,
    root: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for one (corruption, severity) of CIFAR-10-C.

    Returns:
        images: uint8 array, shape (10000, 32, 32, 3), HWC layout, RGB.
        labels: int array, shape (10000,).
    """
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity must be in 1..5, got {severity}")

    root = Path(root) if root is not None else cifar10_c_dir()
    imgs_path = root / f"{corruption}.npy"
    labels_path = root / "labels.npy"
    if not imgs_path.exists():
        raise FileNotFoundError(f"missing CIFAR-10-C array: {imgs_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing CIFAR-10-C labels: {labels_path}")

    images = np.load(imgs_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    start = (severity - 1) * _PER_SEVERITY
    end = severity * _PER_SEVERITY
    return np.asarray(images[start:end]), np.asarray(labels[start:end])
