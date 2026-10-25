"""ImageNet-C loader (Hendrycks's distributed JPEG tree).

After selective extraction by `scripts/data_prep/imagenet_c_download.py`, the
on-disk layout is:

    data/raw/ImageNet-C/<corruption>/<severity>/<wnid>/*.JPEG

Severity ∈ {1..5}. The 50 000 ILSVRC2012 val images are distributed under
their wnids; class index follows torchvision `ImageFolder` convention
(alphabetical wnid sort) — same convention used by the clean ImageNet
ImageFolder built by `scripts/data_prep/imagenet_to_imagefolder.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from torchvision.datasets import ImageFolder

from cbm_in_the_wild.paths import imagenet_c_dir

# Natural-degradation types selected for evaluation.
SELECTED_CORRUPTIONS = ("motion_blur", "gaussian_noise", "jpeg_compression", "fog")


def imagenet_c_severity_dir(
    corruption: str,
    severity: int,
    root: Path | str | None = None,
) -> Path:
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity must be in 1..5, got {severity}")
    root = Path(root) if root is not None else imagenet_c_dir()
    return root / corruption / str(severity)


def load_imagenet_c(
    corruption: str,
    severity: int,
    root: Path | str | None = None,
    transform: Callable | None = None,
) -> ImageFolder:
    """Return a torchvision ImageFolder over (corruption, severity)."""
    sev_dir = imagenet_c_severity_dir(corruption, severity, root=root)
    if not sev_dir.exists():
        raise FileNotFoundError(f"missing ImageNet-C severity dir: {sev_dir}")
    return ImageFolder(str(sev_dir), transform=transform)
