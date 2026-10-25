"""Cosine drift between CLIP embeddings, clean vs corrupted.

Pure tensor math — no CLIP import, no I/O. Works on any L2-normalizable
embedding; we call it "CLIP drift" because that's the pipeline stage it
targets, but the math is generic.

Two shapes supported:
- Single embedding pair: ``(D,)`` vs ``(D,)``.
- Batched: ``(B, D)`` vs ``(B, D)``, paired row-wise.

L2-normalization is applied defensively — DCBM's cached embeddings
are already unit-norm, but re-normalizing is cheap and makes this
robust to callers that pass raw features.
"""

from __future__ import annotations

import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def cosine_drift(clean: torch.Tensor, corrupt: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between paired embeddings.

    Args:
        clean:   ``(D,)`` or ``(B, D)``.
        corrupt: same shape as ``clean``.

    Returns:
        Scalar 0-d tensor for the ``(D,)`` case, or ``(B,)`` for batched.
        Values in [-1, 1]; 1.0 = identical direction.

    "Drift" in the paper's sense is ``1 - cosine``; we return cosine here
    and let callers subtract if they want the distance form. Reporting
    cosine keeps the "clean-vs-clean = 1.0" sanity check obvious.
    """
    if clean.shape != corrupt.shape:
        raise ValueError(
            f"shape mismatch: {tuple(clean.shape)} vs {tuple(corrupt.shape)}"
        )
    a = _l2_normalize(clean)
    b = _l2_normalize(corrupt)
    return (a * b).sum(dim=-1)


def cosine_drift_matrix(
    queries: torch.Tensor, refs: torch.Tensor
) -> torch.Tensor:
    """Cosine similarity matrix between two sets of embeddings.

    Args:
        queries: ``(Nq, D)``.
        refs:    ``(Nr, D)``.

    Returns:
        ``(Nq, Nr)``.

    Useful when corrupt-and-clean segcrops don't correspond 1:1 (SAM2 may
    return a different count of crops on corrupted inputs) and callers
    want to pick nearest-clean for each corrupt crop before scoring.
    """
    if queries.shape[-1] != refs.shape[-1]:
        raise ValueError(
            f"dim mismatch: {queries.shape[-1]} vs {refs.shape[-1]}"
        )
    q = _l2_normalize(queries)
    r = _l2_normalize(refs)
    return q @ r.t()


def nearest_ref_drift(
    corrupt_embeds: torch.Tensor, clean_refs: torch.Tensor
) -> dict:
    """For each corrupt embedding, cosine to its nearest clean reference.

    Handles the unpaired-segcrop case: corrupted SAM2 may produce a
    different number of crops than clean, so row-wise pairing is
    impossible. Match each corrupt crop to its highest-cosine clean
    reference; report per-crop + mean.

    Args:
        corrupt_embeds: ``(Nc, D)``.
        clean_refs:     ``(Nr, D)``.

    Returns:
        ``{"per_item": [float, ...], "mean": float, "n_corrupt": int,
          "n_clean": int}``.
    """
    if corrupt_embeds.numel() == 0 or clean_refs.numel() == 0:
        return {
            "per_item": [],
            "mean": 0.0,
            "n_corrupt": int(corrupt_embeds.shape[0]) if corrupt_embeds.ndim == 2 else 0,
            "n_clean": int(clean_refs.shape[0]) if clean_refs.ndim == 2 else 0,
        }
    m = cosine_drift_matrix(corrupt_embeds, clean_refs)
    best, _ = m.max(dim=1)
    return {
        "per_item": [float(x) for x in best.tolist()],
        "mean": float(best.mean().item()),
        "n_corrupt": int(corrupt_embeds.shape[0]),
        "n_clean": int(clean_refs.shape[0]),
    }
