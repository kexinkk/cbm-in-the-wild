"""DCBM concept-bank helpers: K-means + medoid bank from a segcrop blob, and the
crop key behind each bank centroid.

Same recipe as upstream utils/dcbm.py (cluster_image_concepts / centroid_concepts with
centroid_method='median'): L2-normalise for clustering, take the medoid of the raw
members per cluster, so every concept is one real SAM2 crop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def _medoid(M: np.ndarray) -> np.ndarray:
    """Member L1-nearest the per-dim median (upstream centroid_method='median')."""
    med = np.median(M, axis=0)
    return M[np.abs(M - med).sum(axis=1).argmin()]


def stack_segcrop_emb(blob) -> np.ndarray:
    """(N, D) float32 array of the raw (un-normalised) crop embeddings in a segcrop blob."""
    if isinstance(blob, (str, Path)):
        blob = torch.load(blob, weights_only=False)
    return np.stack([blob[k][0].cpu().numpy() if torch.is_tensor(blob[k][0])
                     else np.asarray(blob[k][0]) for k in blob]).astype(np.float32)


def centroids_from_segcrop(blob, n_clusters: int, *, random_state: int = 42,
                           n_init: int | str = 10) -> torch.Tensor:
    """KMeans over (L2-normalised) crop embeddings → (K, D) medoid bank.

    Mirrors utils/dcbm.py: normalise for clustering, cap K at crop count, take the
    medoid of the *raw* (un-normalised) members per cluster. `blob` may be a segcrop
    dict, a path to one, or an already-stacked (N, D) ndarray (from `stack_segcrop_emb`,
    so the 720 MB blob is loaded once when sweeping many (K, seed) combos).

    `random_state` seeds KMeans (default 42 = the baseline-bank recipe); `n_init` is the
    sklearn restart count (lower = faster, e.g. 3 for a large K sweep).
    """
    emb = blob if isinstance(blob, np.ndarray) else stack_segcrop_emb(blob)
    k = min(n_clusters, emb.shape[0])
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit_predict(normed)
    cents = np.stack([_medoid(emb[labels == c]) for c in range(k)])
    return torch.from_numpy(cents).float()


def _stack_emb(blob, keys=None):
    keys = list(blob) if keys is None else keys
    out = []
    for k in keys:
        e = blob[k][0]
        out.append(e.float() if torch.is_tensor(e) else torch.tensor(e, dtype=torch.float32))
    return torch.stack(out)


def bank_crop_keys(clean_blob, frozen_centroids, *, tol: float = 1.0e-3) -> list[str]:
    """Recover the crop key behind each frozen-bank centroid.

    Because centroid_method='median' (medoid), every centroid is exactly one clean
    crop embedding. We match each centroid to its crop in `clean_blob` by cosine
    (≈1). Returns keys aligned 1:1 with `frozen_centroids` rows. Raises if any
    centroid lacks an exact match (→ frozen bank built from a different segcrop
    snapshot than `clean_blob`).
    """
    keys = list(clean_blob)
    En = F.normalize(_stack_emb(clean_blob, keys), dim=1)
    Cn = F.normalize(frozen_centroids.float(), dim=1)
    sims = Cn @ En.t()                       # (K, N)
    best = sims.argmax(dim=1)
    bestv = sims.gather(1, best.unsqueeze(1)).squeeze(1)
    bad = int((bestv < 1.0 - tol).sum())
    if bad:
        raise ValueError(
            f"{bad}/{len(bestv)} centroids have no exact crop match "
            f"(min cos {bestv.min():.4f}); frozen bank likely from a different "
            f"segcrop snapshot than the given clean_blob"
        )
    return [keys[i] for i in best.tolist()]

