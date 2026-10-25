# -*- coding: utf-8 -*-
"""Hendrycks ImageNet-C corruption generators — algorithmic, 224×224-calibrated.

Vendored from Hendrycks's `robustness` repo:
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py

Reproduces the corruption distribution shipped in the distributed
`data/raw/ImageNet-C/<corruption>/<sev>/<wnid>/*.JPEG` files for our 4 selected
corruptions:
    motion_blur, gaussian_noise, jpeg_compression, fog

Per-severity parameters match Hendrycks's `make_imagenet_c.py` line-for-line.

License: MIT (Hendrycks's robustness repo).

WAND SUBSTITUTION (motion_blur). Same constraint as CIFAR-10 — Wand requires
libmagickwand-dev system-wide; not available on bwHPC. We substitute the
pure-numpy gaussian-weighted line kernel motion blur from bethgelab's
`imagecorruptions` (a faithful Hendrycks port — same algorithm family,
different library). Visual / CLIP-embedding match expected ≥ 0.99; not
byte-identical. Cross-check in
`experiments/post_hoc_adaptation/vendor_verification.json` (Task #15).

Public API:
    apply_hendrycks_imagenet(img_uint8_HW3, corruption, severity, *, seed=None)
        → uint8 (H, W, 3) RGB (H, W as input — 224×224 standard)

    HENDRYCKS_IMAGENET_TRANSFORMS  # {name → callable(PIL_or_ndarray, severity)}
"""
from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from PIL import Image as PILImage

# Re-use shared helpers from the CIFAR-10 vendored module (identical algorithms;
# only the per-severity parameter tables and default plasma_fractal mapsize differ).
from cbm_in_the_wild.corruptions.hendrycks_cifar10 import (
    _motion_blur_numpy,
    _plasma_fractal,
)


def _as_pil(x) -> PILImage.Image:
    if isinstance(x, np.ndarray):
        return PILImage.fromarray(x.astype(np.uint8))
    return x


# =============================================================================
# Corruption functions — Hendrycks ImageNet parameters (224×224)
# =============================================================================

def gaussian_noise(x, severity: int = 1) -> np.ndarray:
    """Hendrycks ImageNet `gaussian_noise`. Sigma ∈ [.08, .12, .18, .26, .38]."""
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def jpeg_compression(x, severity: int = 1) -> np.ndarray:
    """Hendrycks ImageNet `jpeg_compression`. Quality ∈ [25, 18, 15, 10, 7]."""
    c = [25, 18, 15, 10, 7][severity - 1]
    x = _as_pil(x)
    output = BytesIO()
    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)
    return np.array(x.convert("RGB"), dtype=np.uint8)


def fog(x, severity: int = 1) -> np.ndarray:
    """Hendrycks ImageNet `fog`. (intensity, wibbledecay) per-severity table.
    Uses plasma_fractal at mapsize=256 (ImageNet default), slice to image dim."""
    c = [(1.5, 2), (2.0, 2.0), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    x_arr = np.array(x) / 255.0
    max_val = x_arr.max()
    H, W = x_arr.shape[:2]
    # Hendrycks uses mapsize=256 for 224×224 crops. The corrupted-bank pipeline
    # corrupts at NATIVE full resolution (corrupted_bank.py §1), so H or W may
    # exceed 256; _plasma_fractal requires a power-of-two mapsize, so grow it to
    # the next power of two ≥ max(H, W) (≥256 to preserve the 224-default look),
    # then slice [:H, :W].
    mapsize = max(256, 1 << (max(H, W) - 1).bit_length())
    fractal = _plasma_fractal(mapsize=mapsize, wibbledecay=c[1])[:H, :W]
    x_arr = x_arr + c[0] * fractal[..., np.newaxis]
    return np.clip(x_arr * max_val / (max_val + c[0]), 0, 1) * 255


def motion_blur(x, severity: int = 1) -> np.ndarray:
    """Hendrycks ImageNet `motion_blur`. (radius, sigma) per-severity table.

    SUBSTITUTION: pure-numpy implementation (bethgelab style) since Wand is
    not available on bwHPC. Visual / CLIP-embedding match ≥ 0.99 vs distributed
    ImageNet-C `motion_blur/` expected; not byte-identical. Verified by
    Task #15.
    """
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x).astype(np.float32)
    angle = np.random.uniform(-45, 45)
    blurred = _motion_blur_numpy(x, radius=c[0], sigma=c[1], angle=angle)
    return np.clip(blurred, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Dispatch table — matches our `corruptions.imagenet_c.SELECTED_CORRUPTIONS`
# -----------------------------------------------------------------------------
HENDRYCKS_IMAGENET_TRANSFORMS = {
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "fog": fog,
}


def apply_hendrycks_imagenet(
    img: np.ndarray,
    corruption: str,
    severity: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a Hendrycks ImageNet corruption to a clean uint8 (H, W, 3) image
    (typically 224×224 after CenterCrop).

    Seed parameter sets `np.random.seed(...)` before applying — deterministic
    per (image_id, corruption, severity) when callers pass
    `hash((image_id, corruption, severity)) & 0xFFFFFFFF`.

    Returns uint8 (H, W, 3).
    """
    if corruption not in HENDRYCKS_IMAGENET_TRANSFORMS:
        raise ValueError(
            f"unknown Hendrycks ImageNet corruption: {corruption!r}. "
            f"Choices: {list(HENDRYCKS_IMAGENET_TRANSFORMS)}"
        )
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity must be 1..5, got {severity}")
    if seed is not None:
        np.random.seed(int(seed) & 0xFFFFFFFF)
    out = HENDRYCKS_IMAGENET_TRANSFORMS[corruption](img, severity)
    return np.clip(out, 0, 255).astype(np.uint8)
