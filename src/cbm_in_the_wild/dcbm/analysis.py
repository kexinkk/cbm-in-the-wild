"""Analysis primitives — cross-stage attribution + concept-rotation geometry.

Pure functions on already-computed pilot artifacts. No I/O of stage tensors,
no model imports — load tensors / DataFrames at the call site, pass them in.

Three analyses:

1. ``cross_stage_decomposition`` — fit a linear model
   ``acc_drop ~ stage1_drift + stage2_drift`` across cells, return R² + the
   standardised contribution share of each stage. Quantifies whether
   accuracy collapse is CLIP-stage-driven or concept-rank-driven.

2. ``centroid_jump_distances`` — when the top-1 concept rotates from c_A
   to c_B, return ``||c_A - c_B||`` in CLIP-D. Distinguishes near-miss
   rotation (semantically adjacent concepts) from catastrophic scattering.

3. ``per_class_flip_distribution`` — for each ground-truth class, count
   the most common (clean_top1, corrupt_top1) flip pairs. Surfaces whether
   corruption causes within-cluster reordering or cross-class confusion.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# (1) Cross-stage decomposition
# -----------------------------------------------------------------------------
def cross_stage_decomposition(
    df: pd.DataFrame,
    *,
    stage2_metric: str = "argmax_shift_fraction",
    clean_acc_by_dataset: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Regress per-cell accuracy drop on stage-1 + stage-2 drift.

    Args:
        df: long-format frame with at least
            ``dataset, corruption, severity, accuracy, cosine_drift_mean``
            and the chosen stage-2 metric column.
        stage2_metric: one of ``argmax_shift_fraction``, ``top5_jaccard_mean``,
            or ``spearman_rho_mean``. The first one is used directly; the
            other two get converted to drift form (1 − sim).
        clean_acc_by_dataset: per-dataset clean-subset accuracy used as
            ``acc_drop = clean_acc − cell_accuracy``. If None, falls back
            to per-dataset cell-mean — cruder but works for quick looks.

    Returns:
        Frame indexed by dataset with columns
        ``[r2, beta_stage1, beta_stage2, share_stage1, share_stage2, n_cells]``.
        ``share_*`` is the standardised-coefficient absolute share, summing
        to 1.0 per row.
    """
    from sklearn.linear_model import LinearRegression

    metric_to_drift = {
        "argmax_shift_fraction": ("argmax_shift_fraction", lambda s: s),
        "top5_jaccard_mean": ("top5_jaccard_mean", lambda s: 1 - s),
        "spearman_rho_mean": ("spearman_rho_mean", lambda s: 1 - s),
    }
    if stage2_metric not in metric_to_drift:
        raise ValueError(
            f"stage2_metric must be one of {list(metric_to_drift)}, got {stage2_metric!r}"
        )
    s2_col, s2_to_drift = metric_to_drift[stage2_metric]

    out = []
    for ds, sub in df.groupby("dataset"):
        if clean_acc_by_dataset and ds in clean_acc_by_dataset:
            clean = clean_acc_by_dataset[ds]
        else:
            clean = sub.accuracy.max()  # crude fallback
        s1_drift = (1 - sub["cosine_drift_mean"]).to_numpy()
        s2_drift = s2_to_drift(sub[s2_col]).to_numpy()
        y = (clean - sub.accuracy).to_numpy()
        X = np.column_stack([s1_drift, s2_drift])
        m = LinearRegression().fit(X, y)
        b1, b2 = float(m.coef_[0]), float(m.coef_[1])
        s1_share = abs(b1 * s1_drift.std() / y.std()) if y.std() > 0 else 0.0
        s2_share = abs(b2 * s2_drift.std() / y.std()) if y.std() > 0 else 0.0
        total = s1_share + s2_share if (s1_share + s2_share) > 0 else 1.0
        out.append({
            "dataset": ds,
            "stage2_metric": stage2_metric,
            "r2": float(m.score(X, y)),
            "beta_stage1": b1,
            "beta_stage2": b2,
            "share_stage1": s1_share / total,
            "share_stage2": s2_share / total,
            "n_cells": int(len(sub)),
        })
    return pd.DataFrame(out).set_index("dataset")


# -----------------------------------------------------------------------------
# (2) Centroid-jump distances
# -----------------------------------------------------------------------------
def centroid_jump_distances(
    centroids: torch.Tensor,
    clean_top1: torch.Tensor,
    corrupt_top1: torch.Tensor,
) -> torch.Tensor:
    """Per-image L2 distance ``||c_clean - c_corrupt||`` in centroid space.

    Args:
        centroids:    ``(K, D)`` centroid bank.
        clean_top1:   ``(B,)`` int — top-1 concept index per image, clean side.
        corrupt_top1: ``(B,)`` int — top-1 concept index per image, corrupted side.

    Returns:
        ``(B,)`` float distances. Zero where the top-1 concept is unchanged.
    """
    if clean_top1.shape != corrupt_top1.shape:
        raise ValueError(
            f"shape mismatch: {tuple(clean_top1.shape)} vs {tuple(corrupt_top1.shape)}"
        )
    c_a = centroids[clean_top1.long()]
    c_b = centroids[corrupt_top1.long()]
    return (c_a - c_b).norm(dim=-1)


def centroid_pairwise_summary(centroids: torch.Tensor) -> dict:
    """Distribution of off-diagonal pairwise distances + nearest-neighbour stats.

    Useful as a reference scale for ``centroid_jump_distances`` interpretation:
    a jump near the nearest-neighbour median means within-cluster rotation;
    a jump near the off-diagonal median means random scattering.
    """
    K = centroids.shape[0]
    D = torch.cdist(centroids, centroids)
    mask = ~torch.eye(K, dtype=bool, device=D.device)
    off = D[mask]
    nn = D.masked_fill(~mask, float("inf")).min(dim=1).values
    return {
        "K": int(K),
        "off_diag_min": float(off.min()),
        "off_diag_median": float(off.median()),
        "off_diag_mean": float(off.mean()),
        "off_diag_max": float(off.max()),
        "nn_median": float(nn.median()),
        "nn_mean": float(nn.mean()),
    }


# -----------------------------------------------------------------------------
# (3) Per-class flip distribution
# -----------------------------------------------------------------------------
def per_class_flip_distribution(
    labels: torch.Tensor,
    clean_top1: torch.Tensor,
    corrupt_top1: torch.Tensor,
    *,
    top_k: int = 3,
) -> dict[int, list[tuple[int, int, int]]]:
    """For each ground-truth class, the ``top_k`` most common (clean→corrupt) pairs.

    Args:
        labels:       ``(B,)`` ground-truth class index per image.
        clean_top1:   ``(B,)`` clean top-1 concept index.
        corrupt_top1: ``(B,)`` corrupted top-1 concept index.
        top_k: how many most-common flip pairs to return per class.

    Returns:
        ``{class_idx: [(clean_concept, corrupt_concept, count), ...]}``,
        sorted by count descending. Identity flips (where the top-1 didn't
        change) are included — caller can filter on
        ``clean_concept == corrupt_concept`` if needed.
    """
    out: dict[int, list[tuple[int, int, int]]] = {}
    labels_l = labels.long().tolist()
    clean_l = clean_top1.long().tolist()
    corrupt_l = corrupt_top1.long().tolist()
    by_class: dict[int, Counter] = defaultdict(Counter)
    for lab, c, p in zip(labels_l, clean_l, corrupt_l):
        by_class[lab][(c, p)] += 1
    for cls, ctr in by_class.items():
        items = sorted(ctr.items(), key=lambda kv: -kv[1])[:top_k]
        out[cls] = [(c, p, n) for (c, p), n in items]
    return out


# -----------------------------------------------------------------------------
# Loaders for per-image dumps from `propagate_corrupted_cell(per_image_dir=...)`
# -----------------------------------------------------------------------------
def load_per_image_cell(per_image_dir: Path, corruption: str, severity: int) -> dict:
    """Read one cell's per-image dump (`<corruption>_s<sev>.pt`).

    Returns a dict with `image_ids, labels, clean_top1_concept,
    corrupt_top1_concept, clean_pred, corrupt_pred`. Keys mirror the file
    schema written by `pipeline.propagate_corrupted_cell`.
    """
    path = Path(per_image_dir) / f"{corruption}_s{severity}.pt"
    if not path.exists():
        raise FileNotFoundError(f"per-image dump missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
