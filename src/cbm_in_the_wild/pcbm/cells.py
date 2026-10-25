"""Corrupted test-split CLIP-embedding cells for the PCBM ablation.

ImageNet: reuse the 50 full-test cells already cached by
experiments/imagenet_c_propagation/<fam>_s<sev>/per_image/ (clip_emb key).
PCBM runs on the same CLIP ViT-B/16 space, so these are reused verbatim with
zero new corrupted-image compute — identical to the LF-CBM point.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cbm_in_the_wild.paths import REPO_ROOT

IMAGENET_PROP_ROOT = REPO_ROOT / "experiments" / "imagenet_c_propagation"


def cell_path(cfg: dict, exp_dir: Path, corruption: str, severity: int) -> Path:
    if cfg["dataset"] == "imagenet":
        d = IMAGENET_PROP_ROOT / f"{corruption}_s{severity}" / "per_image"
        return d / f"{corruption}_s{severity}.pt"
    # Optional shared cache: the corrupted CLIP embedding z(x) is backbone-only
    # and identical for LF-CBM and PCBM (both CLIP ViT-B/16).
    ctd = cfg.get("corrupted_test_dir")
    if ctd:
        base = Path(ctd)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / f"{corruption}_s{severity}.pt"
    return exp_dir / "corrupted_test" / f"{corruption}_s{severity}.pt"


def load_cell(cfg: dict, exp_dir: Path, corruption: str, severity: int) -> dict:
    """→ {"clip_emb": [N,512] f32, "image_ids": list[str]}"""
    blob = torch.load(cell_path(cfg, exp_dir, corruption, severity), map_location="cpu")
    return {"clip_emb": blob["clip_emb"].float(), "image_ids": list(blob["image_ids"])}
