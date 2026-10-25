# -*- coding: utf-8 -*-
"""Hendrycks CIFAR-10-C corruption generators — algorithmic, 32×32-calibrated.

Vendored from Hendrycks's `robustness` repo:
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py

Reproduces the exact corruption distribution shipped in the distributed
`data/raw/CIFAR-10-C/<corruption>.npy` files for our 4 selected corruptions:
    motion_blur, gaussian_noise, jpeg_compression, fog

Per-severity parameters match Hendrycks's `make_cifar_c.py` line-for-line.

License: MIT (Hendrycks's robustness repo).

WAND SUBSTITUTION (motion_blur). Hendrycks's original motion_blur uses Wand
(Python ImageMagick wrapper) which requires libmagickwand-dev system-wide;
not available on bwHPC. We substitute the pure-numpy gaussian-weighted line
kernel motion blur from bethgelab's `imagecorruptions` (a faithful Hendrycks
port — same algorithm family, different library). Visual / CLIP-embedding
match expected ≥ 0.99; not byte-identical. Cross-check in
`experiments/post_hoc_adaptation/vendor_verification.json` (Task #15).

Public API:
    apply_hendrycks_cifar10(img_uint8_HW3, corruption, severity, *, seed=None)
        → uint8 (32, 32, 3) RGB

    HENDRYCKS_CIFAR10_TRANSFORMS  # {name → callable(PIL_image, severity)}
"""
from __future__ import annotations

import math
from io import BytesIO

import cv2
import numpy as np
from PIL import Image as PILImage
from skimage.filters import gaussian


# -----------------------------------------------------------------------------
# Plasma fractal — verbatim from Hendrycks make_cifar_c.py (default mapsize=32
# matches CIFAR-10 image size).
# Source: github.com/FLHerne/mapgen/blob/master/diamondsquare.py (per Hendrycks)
# -----------------------------------------------------------------------------
def _plasma_fractal(mapsize: int = 32, wibbledecay: float = 3) -> np.ndarray:
    """Diamond-square heightmap. Returns (mapsize, mapsize) float in [0, 1]."""
    assert (mapsize & (mapsize - 1) == 0), "mapsize must be a power of two"
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        ms = maparray.shape[0]
        drgrid = maparray[stepsize // 2:ms:stepsize, stepsize // 2:ms:stepsize]
        ulgrid = maparray[0:ms:stepsize, 0:ms:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:ms:stepsize, stepsize // 2:ms:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:ms:stepsize, 0:ms:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


# -----------------------------------------------------------------------------
# Motion-blur kernel helpers — bethgelab pure-numpy substitute for Wand.
# Source: bethgelab/imagecorruptions imagecorruptions/corruptions.py
# (faithful Hendrycks port without the ImageMagick C dep)
# -----------------------------------------------------------------------------
def _gauss_function(x, mean, sigma):
    return (np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)


def _motion_blur_kernel(width: int, sigma: float) -> np.ndarray:
    k = _gauss_function(np.arange(width), 0, sigma)
    return k / k.sum()


def _shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1:dx]
    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx:dx + 1]
    else:
        shifted = image
    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1:dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy:dy + 1, :]
    return shifted


def _motion_blur_numpy(x: np.ndarray, radius: float, sigma: float, angle: float) -> np.ndarray:
    """Pure-numpy motion blur — bethgelab/imagecorruptions style."""
    width = int(radius * 2 + 1)
    kernel = _motion_blur_kernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]:
            break
        shifted = _shift_image(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred


# =============================================================================
# Corruption functions — Hendrycks CIFAR-10 parameters (32×32)
# =============================================================================
# All `x` inputs accept either uint8 (32, 32, 3) numpy or PIL.Image. Each
# returns uint8 (32, 32, 3) numpy in [0, 255]. RNG: `np.random` global state —
# callers seed via `np.random.seed(...)` before invoking for reproducibility.
# -----------------------------------------------------------------------------

def _as_pil(x) -> PILImage.Image:
    if isinstance(x, np.ndarray):
        return PILImage.fromarray(x.astype(np.uint8))
    return x


def gaussian_noise(x, severity: int = 1) -> np.ndarray:
    """Hendrycks CIFAR `gaussian_noise`. Severity sigma ∈ [.04, .06, .08, .09, .10]."""
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def jpeg_compression(x, severity: int = 1) -> np.ndarray:
    """Hendrycks CIFAR `jpeg_compression`. Quality ∈ [80, 65, 58, 50, 40]."""
    c = [80, 65, 58, 50, 40][severity - 1]
    x = _as_pil(x)
    output = BytesIO()
    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)
    return np.array(x.convert("RGB"), dtype=np.uint8)


def fog(x, severity: int = 1) -> np.ndarray:
    """Hendrycks CIFAR `fog`. (intensity, wibbledecay) per-severity table.
    Uses plasma_fractal at mapsize=32 (CIFAR-native)."""
    c = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1.0, 2.0), (1.5, 1.75)][severity - 1]
    x = np.array(x) / 255.0
    max_val = x.max()
    x = x + c[0] * _plasma_fractal(mapsize=32, wibbledecay=c[1])[:32, :32][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def motion_blur(x, severity: int = 1) -> np.ndarray:
    """Hendrycks CIFAR `motion_blur`. (radius, sigma) per-severity table.

    SUBSTITUTION: pure-numpy implementation (bethgelab style) since Wand is
    not available on bwHPC. Visual / CLIP-embedding match ≥ 0.99 vs distributed
    CIFAR-10-C `motion_blur.npy` expected; not byte-identical. Verified by
    Task #15.
    """
    c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]
    x = np.array(x).astype(np.float32)
    angle = np.random.uniform(-45, 45)
    blurred = _motion_blur_numpy(x, radius=c[0], sigma=c[1], angle=angle)
    return np.clip(blurred, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Dispatch table — matches our `corruptions.cifar10c.SELECTED_CORRUPTIONS`
# -----------------------------------------------------------------------------
HENDRYCKS_CIFAR10_TRANSFORMS = {
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "fog": fog,
}


def apply_hendrycks_cifar10(
    img: np.ndarray,
    corruption: str,
    severity: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a Hendrycks CIFAR-10 corruption to a clean uint8 (32, 32, 3) image.

    Seed parameter sets `np.random.seed(...)` before applying — deterministic
    per (image_id, corruption, severity) when callers pass
    `hash((image_id, corruption, severity)) & 0xFFFFFFFF`.

    Returns uint8 (32, 32, 3).
    """
    if corruption not in HENDRYCKS_CIFAR10_TRANSFORMS:
        raise ValueError(
            f"unknown Hendrycks CIFAR-10 corruption: {corruption!r}. "
            f"Choices: {list(HENDRYCKS_CIFAR10_TRANSFORMS)}"
        )
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity must be 1..5, got {severity}")
    if seed is not None:
        np.random.seed(int(seed) & 0xFFFFFFFF)
    out = HENDRYCKS_CIFAR10_TRANSFORMS[corruption](img, severity)
    return np.clip(out, 0, 255).astype(np.uint8)
