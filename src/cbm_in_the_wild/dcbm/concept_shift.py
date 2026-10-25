"""Bank-relative per-concept corruption shift ``d_k`` — reusable streaming pass.

``d_k`` (supervisor's metric, decision 2026-06-30) is defined on the pre-softmax
per-concept score ``raw_k(z) = z·C_kᵀ/‖C_k‖²``:

    d_k = nanmean over (family, severity) of  mean_i |raw_k(corrupt_i) − raw_k(clean_i)|

It is **bank-relative** — each concept's score depends only on its own centroid row
``C_k`` (the scores are per-concept independent), so ``d_k`` can be recomputed for any
bank, and for an intervention that touches only a few rows, only those rows change.

This mirrors the streaming pass in
``experiments/imagenet_concept_shift_robust_set/run.py`` (the §3.4 producer) but is
factored here so the prune-source ablation (T3, 800-dim banks) and the crop-quality
causal test (T1b, spliced 2048-dim bank) share one definition. ``recompute_dk`` scores
several banks in a single pass over the corrupt cells, so the ~25 GB of ``torch.load``
is paid once regardless of how many banks are ranked.

Not imported by ``run.py`` — that runner is left untouched; this is for the follow-ups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from cbm_in_the_wild.paths import CHECKPOINT_DIR, REPO_ROOT
from cbm_in_the_wild.dcbm.analysis import load_per_image_cell

# The 10 ImageNet-C families the §3.4 pass uses (order fixed; matches shift_grids.npz).
FAMILIES = ["fog", "gaussian_noise", "motion_blur", "patch_masking", "salt_and_pepper",
            "jpeg_compression", "row_column_dropout", "random_pixel_dropout",
            "sparse_random_occlusion", "structured_pixel_dropout"]
SEVERITIES = [1, 2, 3, 4, 5]

PROP_ROOT = REPO_ROOT / "experiments/imagenet_c_propagation"
CLEAN_DIR = CHECKPOINT_DIR / "imagenet_baseline" / "clean_propagation"


def load_clean_ref() -> tuple[torch.Tensor, dict]:
    """(clean CLIP embeddings (N, D), {image_id: row}) for the full 50k test split."""
    emb = torch.load(CLEAN_DIR / "clip_emb.pt", map_location="cpu").float()
    ids = torch.load(CLEAN_DIR / "image_ids.pt", map_location="cpu")
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    return emb, {iid: i for i, iid in enumerate(ids)}


def recompute_dk(
    banks: Sequence[torch.Tensor],
    *,
    exclude_sev: int | None = None,
    families: Sequence[str] = FAMILIES,
    severities: Sequence[int] = SEVERITIES,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Per-concept ``d_k`` for each bank in ``banks`` (one (K_i,) array each).

    Streams every ``(family, severity)`` corrupt cell once and scores all banks against
    it, so multi-bank ranking costs a single I/O pass. ``exclude_sev`` (e.g. 3) drops
    that severity from the mean → leakage-free ranking when the eval severity is 3.
    """
    banks = [b.float() for b in banks]
    normsq = [(b ** 2).sum(1) for b in banks]                      # (K_i,) per bank
    sevs = [s for s in severities if s != exclude_sev]

    CLEAN_EMB, POS = load_clean_ref()

    def raw(z: torch.Tensor, bi: int) -> torch.Tensor:
        return (z @ banks[bi].t()) / normsq[bi]                    # (n, K_i)

    grids = [np.full((b.shape[0], len(families), len(sevs)), np.nan) for b in banks]
    for fi, fam in enumerate(families):
        for si, sev in enumerate(sevs):
            pdir = PROP_ROOT / f"{fam}_s{sev}" / "per_image"
            try:
                cell = load_per_image_cell(pdir, fam, sev)
            except FileNotFoundError:
                if verbose:
                    print(f"  skip missing {fam}_s{sev}", flush=True)
                continue
            cids = cell["image_ids"]
            cids = cids.tolist() if isinstance(cids, torch.Tensor) else list(cids)
            idx = torch.tensor([POS[i] for i in cids], dtype=torch.long)
            cl_e = CLEAN_EMB.index_select(0, idx)
            ce = cell["clip_emb"].float()
            for bi in range(len(banks)):
                d = (raw(ce, bi) - raw(cl_e, bi)).abs().mean(0).numpy()
                grids[bi][:, fi, si] = d
            del cell, ce, cl_e
        if verbose:
            print(f"  d_k cells done: {fam}", flush=True)

    return [np.nanmean(g, axis=(1, 2)) for g in grids]
