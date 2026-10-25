"""Custom sensor-level corruption transforms.

Six transforms simulating physical sensor faults: random and structured
pixel dropout, salt-and-pepper noise, sparse random occlusion, single-
patch masking, and row/column dropout.

All transforms operate on torch tensors of shape (..., H, W) — typically
(C, H, W) post-`ToTensor()` — with dtype float32 in [0, 1] or uint8 in
[0, 255]. The same mask is applied across channels, so RGB imagery stays
color-consistent.

Severities ∈ {1..5} monotonically increase information loss (sev 1
minimal-but-measurable, sev 5 severe but recoverable). Randomness is
seedable via an explicit `torch.Generator`; without one, the global
torch RNG is used.

Each transform is a callable class compatible with torchvision `Compose`.
"""

from __future__ import annotations

import torch
from torch import Tensor

_VALID_SEVERITIES = (1, 2, 3, 4, 5)


def _check_severity(severity: int) -> None:
    if severity not in _VALID_SEVERITIES:
        raise ValueError(f"severity must be in {_VALID_SEVERITIES}, got {severity}")


def _white_value(img: Tensor) -> float:
    return 255.0 if img.dtype == torch.uint8 else 1.0


def _randint(
    high: int,
    size: tuple[int, ...],
    generator: torch.Generator | None,
) -> Tensor:
    return torch.randint(0, high, size, generator=generator)


class RandomPixelDropout:
    """Sparse 'dead' pixels — each pixel independently zeroed w.p. p.

    Simulates CMOS/CCD sparse pixel failure. Fraction dropped per severity:
    ``(0.01, 0.03, 0.07, 0.13, 0.22)``. Geometric step ~2× per severity;
    sev 5 calibrated to land near PSNR 13 / SSIM 0.32 on natural images,
    keeping classifier accuracy above the random-guess floor.
    """

    PROB = (0.01, 0.03, 0.07, 0.13, 0.22)

    def __init__(self, severity: int, generator: torch.Generator | None = None):
        _check_severity(severity)
        self.severity = severity
        self.p = self.PROB[severity - 1]
        self.generator = generator

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        keep = (torch.rand(h, w, generator=self.generator) >= self.p).to(img.dtype)
        return img * keep


class SaltAndPepperNoise:
    """Impulsive noise — p fraction of pixels forced to 0 or max, 50/50.

    Simulates bit errors / hot & dead pixels. Fraction affected per
    severity: ``(0.01, 0.03, 0.07, 0.13, 0.22)``.
    """

    PROB = (0.01, 0.03, 0.07, 0.13, 0.22)

    def __init__(self, severity: int, generator: torch.Generator | None = None):
        _check_severity(severity)
        self.severity = severity
        self.p = self.PROB[severity - 1]
        self.generator = generator

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        r = torch.rand(h, w, generator=self.generator)
        keep = (r >= self.p).to(img.dtype)
        salt = ((r >= self.p / 2) & (r < self.p)).to(img.dtype)
        # Pepper positions fall out: keep=0 and salt=0 → pixel becomes 0.
        return img * keep + _white_value(img) * salt


class SparseRandomOcclusion:
    """Many small irregular-shape occlusions — dust / droplet simulacrum.

    N small rectangles with random aspect ratios, each zeroed out; total
    occluded area ≈ ``(0.02, 0.05, 0.10, 0.18, 0.30)[s-1]`` of the image.
    Per-blob area held roughly constant (~3 px² on 32×32, ~150 px² on 224×224)
    by scaling N_BLOBS with AREA.
    """

    AREA = (0.02, 0.05, 0.10, 0.18, 0.30)
    N_BLOBS = (8, 15, 30, 60, 100)

    def __init__(self, severity: int, generator: torch.Generator | None = None):
        _check_severity(severity)
        self.severity = severity
        self.area = self.AREA[severity - 1]
        self.n = self.N_BLOBS[severity - 1]
        self.generator = generator

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        per_blob_area = (self.area * h * w) / self.n
        # Random aspect ratio ∈ [0.5, 2.0] per blob, solved for (bh, bw).
        ratios = 0.5 + 1.5 * torch.rand(self.n, generator=self.generator)
        bh = torch.clamp((per_blob_area / ratios).sqrt().round().long(), min=1, max=h)
        bw = torch.clamp((per_blob_area * ratios).sqrt().round().long(), min=1, max=w)
        ys = _randint(h, (self.n,), self.generator)
        xs = _randint(w, (self.n,), self.generator)
        mask = torch.ones(h, w, dtype=img.dtype)
        for i in range(self.n):
            y0, x0 = ys[i].item(), xs[i].item()
            y1 = min(h, y0 + bh[i].item())
            x1 = min(w, x0 + bw[i].item())
            mask[y0:y1, x0:x1] = 0
        return img * mask


class PatchMasking:
    """Single large contiguous rectangular patch (near-square aspect).

    Patch area ≈ ``(0.02, 0.05, 0.10, 0.18, 0.30)[s-1]`` of the image;
    position uniform random within valid bounds.
    """

    AREA = (0.02, 0.05, 0.10, 0.18, 0.30)

    def __init__(self, severity: int, generator: torch.Generator | None = None):
        _check_severity(severity)
        self.severity = severity
        self.area = self.AREA[severity - 1]
        self.generator = generator

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        side = int(round((self.area * h * w) ** 0.5))
        side = max(1, min(side, min(h, w)))
        y0 = _randint(h - side + 1, (1,), self.generator).item()
        x0 = _randint(w - side + 1, (1,), self.generator).item()
        mask = torch.ones(h, w, dtype=img.dtype)
        mask[y0 : y0 + side, x0 : x0 + side] = 0
        return img * mask


class RowColumnDropout:
    """Whole-row or whole-column dropout — scanline loss.

    Fraction dropped per severity: ``(0.02, 0.05, 0.10, 0.17, 0.25)``.
    Chooses rows or columns 50/50 at construction time (seedable).
    """

    FRAC = (0.02, 0.05, 0.10, 0.17, 0.25)

    def __init__(self, severity: int, generator: torch.Generator | None = None):
        _check_severity(severity)
        self.severity = severity
        self.frac = self.FRAC[severity - 1]
        self.generator = generator

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        # 50/50 pick between rows and columns, via the generator.
        axis = _randint(2, (1,), self.generator).item()  # 0 = rows, 1 = cols
        dim = h if axis == 0 else w
        k = max(1, int(round(self.frac * dim)))
        idx = torch.randperm(dim, generator=self.generator)[:k]
        mask = torch.ones(h, w, dtype=img.dtype)
        if axis == 0:
            mask[idx, :] = 0
        else:
            mask[:, idx] = 0
        return img * mask


class StructuredPixelDropout:
    """Regular Bayer-like grid pattern — deterministic, seed-invariant.

    Drop fractions per severity (exact by construction): 1/64, 1/32, 1/16,
    1/8, 1/4 — geometric ladder via single-corner-drop tiles. Deterministic
    given (severity, H, W). Even sev 1 (1/64) lands near PSNR 24; structured
    drops can't reach Hendrycks-level "barely visible" without enormous
    tiles, since spatial correlation dominates SSIM at any drop fraction.
    """

    # Tile masks: bottom-right cell zeroed, rest kept. 1=keep, 0=drop.
    PATTERNS = {
        1: tuple([(1,) * 8] * 7) + ((1,) * 7 + (0,),),   # 8x8, drop 1/64
        2: tuple([(1,) * 8] * 3) + ((1,) * 7 + (0,),),   # 4x8, drop 1/32
        3: tuple([(1,) * 4] * 3) + ((1,) * 3 + (0,),),   # 4x4, drop 1/16
        4: ((1, 1), (1, 1), (1, 1), (1, 0)),             # 4x2, drop 1/8
        5: ((1, 1), (1, 0)),                             # 2x2, drop 1/4
    }

    def __init__(self, severity: int, generator: torch.Generator | None = None):
        _check_severity(severity)
        self.severity = severity
        # generator accepted for API symmetry; ignored (pattern is deterministic).
        self.generator = generator

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        tile = torch.tensor(self.PATTERNS[self.severity], dtype=img.dtype)
        th, tw = tile.shape
        reps_h = (h + th - 1) // th
        reps_w = (w + tw - 1) // tw
        mask = tile.repeat(reps_h, reps_w)[:h, :w]
        return img * mask


# Ordered registry.
SENSOR_TRANSFORMS = {
    "random_pixel_dropout": RandomPixelDropout,
    "salt_and_pepper": SaltAndPepperNoise,
    "sparse_random_occlusion": SparseRandomOcclusion,
    "patch_masking": PatchMasking,
    "row_column_dropout": RowColumnDropout,
    "structured_pixel_dropout": StructuredPixelDropout,
}

__all__ = [
    "RandomPixelDropout",
    "SaltAndPepperNoise",
    "SparseRandomOcclusion",
    "PatchMasking",
    "RowColumnDropout",
    "StructuredPixelDropout",
    "SENSOR_TRANSFORMS",
]
