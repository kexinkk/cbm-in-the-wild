"""Corrupted ImageNet *train* images for the disjoint d_k pool.

The disjoint pool (experiments/imagenet_dk_disjoint_pool) re-estimates d_k on 50
ImageNet train images per class, disjoint from the 50k evaluation images and the
128,118-image refit pool. Each image is loaded with Resize(256) + CenterCrop(224)
(the test-side preprocessing) and corrupted with the per-image seed
``hash((image_id, corruption, severity)) & 0xFFFFFFFF``. Python string hashing is
salted per process, so launch with ``PYTHONHASHSEED=0`` to reproduce the released
realization.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from cbm_in_the_wild.corruptions import (
    IMAGENET_C_SELECTED,
    SENSOR_TRANSFORMS,
    apply_hendrycks_imagenet,
)
from cbm_in_the_wild.paths import imagenet_image_path


def corruption_seed(image_id: str, corruption: str, severity: int) -> int:
    return hash((image_id, corruption, severity)) & 0xFFFFFFFF


def apply_corruption(img_uint8_HW3: np.ndarray, corruption: str, severity: int,
                     *, seed: int | None = None) -> np.ndarray:
    """One of the 10 corruptions on a 224x224 uint8 image.

    The 4 natural types use the vendored Hendrycks ImageNet-C transforms; the 6
    sensor types use SENSOR_TRANSFORMS with a seeded torch generator.
    """
    if corruption in IMAGENET_C_SELECTED:
        return apply_hendrycks_imagenet(img_uint8_HW3, corruption, severity, seed=seed)
    if corruption in SENSOR_TRANSFORMS:
        t = torch.from_numpy(img_uint8_HW3).permute(2, 0, 1)
        gen = torch.Generator().manual_seed((seed or 42) & 0xFFFFFFFF)
        out = SENSOR_TRANSFORMS[corruption](severity=severity, generator=gen)(t)
        return out.permute(1, 2, 0).numpy().astype(np.uint8)
    raise ValueError(f"unknown corruption: {corruption!r}. "
                     f"Choices: {sorted(set(IMAGENET_C_SELECTED) | set(SENSOR_TRANSFORMS))}")


def load_train_image(image_id: str, wnid: str) -> np.ndarray:
    """ImageNet train image -> Resize(256) + CenterCrop(224) uint8 HxWx3."""
    from torchvision import transforms as T
    cc = T.Compose([T.Resize(256), T.CenterCrop(224)])
    with Image.open(imagenet_image_path("train", wnid, image_id)) as im:
        return np.asarray(cc(im.convert("RGB")))


def corrupted_train_image(image_id: str, wnid: str, corruption: str, severity: int) -> np.ndarray:
    return apply_corruption(load_train_image(image_id, wnid), corruption, severity,
                            seed=corruption_seed(image_id, corruption, severity))
