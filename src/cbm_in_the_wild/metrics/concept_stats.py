"""Concept-activation statistics.

Concept activations in DCBM are the ``(B, K)`` matrix produced by
projecting CLIP image embeddings onto ``K`` concept centroids (cosine
similarity, per `CBM.compute_concept_activations`). This module scores
stability + spread of that matrix between two passes (e.g. clean vs
corrupted): per-image cosine, top-k overlap, softmax entropy, and the
paired entropy delta. Together these distinguish "concept signal
disappeared / distribution flattened" (higher entropy, similar cosine)
from "concept signal redirected to different concepts" (lower cosine,
similar entropy).
"""

from __future__ import annotations

import torch

from cbm_in_the_wild.metrics.clip_drift import cosine_drift


def per_concept_cosine(
    clean_acts: torch.Tensor, corrupt_acts: torch.Tensor
) -> torch.Tensor:
    """Cosine between clean and corrupt concept-activation vectors.

    Args:
        clean_acts:   ``(B, K)``.
        corrupt_acts: ``(B, K)``.

    Returns:
        ``(B,)`` — per-image cosine between the full K-dim activation
        vectors. 1.0 = activations identical after renormalization;
        lower = concept signal shifted.
    """
    if clean_acts.shape != corrupt_acts.shape:
        raise ValueError(
            f"activation shape mismatch: "
            f"{tuple(clean_acts.shape)} vs {tuple(corrupt_acts.shape)}"
        )
    return cosine_drift(clean_acts, corrupt_acts)


def softmax_entropy(acts: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Shannon entropy of the softmax distribution over concepts.

    Args:
        acts: ``(B, K)`` activation scores (cosines, logits — whatever).
        temperature: softmax temperature; default 1.0. Higher = flatter.

    Returns:
        ``(B,)`` entropy in nats. Range is ``[0, log(K)]`` — 0 when one
        concept dominates, ``log(K)`` when activations are uniform.

    Temperature note: DCBM's concept activations are cosines in ``[-1, 1]``
    and their softmax is quite peaked even before temperature scaling,
    so the default ``T=1`` is the right thing for "raw" entropy. Use
    higher ``T`` to make the metric more sensitive to mid-range shifts
    if that's needed for a specific analysis.
    """
    if acts.ndim != 2:
        raise ValueError(f"expected (B, K), got {tuple(acts.shape)}")
    logp = torch.log_softmax(acts / temperature, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def entropy_shift(
    clean_acts: torch.Tensor,
    corrupt_acts: torch.Tensor,
    temperature: float = 1.0,
) -> dict:
    """Paired entropy before / after, with signed delta.

    Read together with ``per_concept_cosine``:
        - Δentropy > 0, cosine ≈ 1 → concept weights flattened.
        - Δentropy ≈ 0, cosine ↓   → concept weights redirected.

    Args:
        clean_acts:   ``(B, K)``.
        corrupt_acts: ``(B, K)``.

    Returns:
        ``{"clean_entropy": (B,), "corrupt_entropy": (B,),
          "delta_entropy": (B,), "cosine": (B,)}`` — all float tensors.
    """
    h_clean = softmax_entropy(clean_acts, temperature=temperature)
    h_corrupt = softmax_entropy(corrupt_acts, temperature=temperature)
    cos = per_concept_cosine(clean_acts, corrupt_acts)
    return {
        "clean_entropy": h_clean,
        "corrupt_entropy": h_corrupt,
        "delta_entropy": h_corrupt - h_clean,
        "cosine": cos,
    }


def top_k_concept_overlap(
    clean_acts: torch.Tensor, corrupt_acts: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Jaccard overlap of the top-``k`` concepts per image.

    Complements the cosine + entropy signature: catches the case where
    two activation vectors have similar shape (high cosine) but the
    top-ranked concepts rotate (low overlap).

    Args:
        clean_acts:   ``(B, K)``.
        corrupt_acts: ``(B, K)``.
        k: top-k cutoff; defaults to 5.

    Returns:
        ``(B,)`` Jaccard in ``[0, 1]``.
    """
    if clean_acts.shape != corrupt_acts.shape:
        raise ValueError(
            f"shape mismatch: {tuple(clean_acts.shape)} vs {tuple(corrupt_acts.shape)}"
        )
    _, top_c = clean_acts.topk(k, dim=-1)
    _, top_p = corrupt_acts.topk(k, dim=-1)
    out = torch.empty(clean_acts.shape[0], dtype=torch.float32)
    for i in range(clean_acts.shape[0]):
        sa = set(top_c[i].tolist())
        sb = set(top_p[i].tolist())
        union = len(sa | sb)
        out[i] = 0.0 if union == 0 else len(sa & sb) / union
    return out


def argmax_shift_fraction(
    clean_acts: torch.Tensor, corrupt_acts: torch.Tensor
) -> float:
    """Fraction of images where the top-1 concept index differs clean vs corrupted.

    Paired comparison at a single severity — not a continuum, so
    "fraction" rather than "rate"; "flip" / mFR terminology is reserved
    for perturbation in this codebase.
    The hardest version of top-k Jaccard (k=1) reduced to a scalar.
    Discriminative when the dominant concept rotates even though the
    top-5 set holds — empirically the strongest stage-2 signal under
    DCBM's near-uniform activation geometry.

    Args:
        clean_acts:   ``(B, K)`` post-softmax activations.
        corrupt_acts: ``(B, K)`` post-softmax activations.

    Returns:
        Scalar in ``[0, 1]``: fraction of images whose top-1 concept shifted.
    """
    if clean_acts.shape != corrupt_acts.shape:
        raise ValueError(
            f"shape mismatch: {tuple(clean_acts.shape)} vs {tuple(corrupt_acts.shape)}"
        )
    shifted = clean_acts.argmax(dim=-1) != corrupt_acts.argmax(dim=-1)
    return float(shifted.float().mean().item())


def kl_divergence(
    clean_acts: torch.Tensor,
    corrupt_acts: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-image ``D_KL(clean || corrupt)`` of the K-dim softmax.

    Catches mass redistribution that cosine + entropy miss: when corruption
    moves probability between concepts of similar magnitude, the shape +
    entropy stay constant but KL is non-zero. Asymmetric — clean is the
    reference distribution.

    Args:
        clean_acts:   ``(B, K)`` post-softmax activations.
        corrupt_acts: ``(B, K)`` post-softmax activations.
        eps: floor for numerical stability under log.

    Returns:
        ``(B,)`` KL divergence in nats. Non-negative; 0 iff clean == corrupt.
    """
    if clean_acts.shape != corrupt_acts.shape:
        raise ValueError(
            f"shape mismatch: {tuple(clean_acts.shape)} vs {tuple(corrupt_acts.shape)}"
        )
    p = clean_acts.clamp_min(eps)
    q = corrupt_acts.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def spearman_rho(
    clean_acts: torch.Tensor, corrupt_acts: torch.Tensor
) -> torch.Tensor:
    """Per-image Spearman rank correlation between K-dim activation vectors.

    Pearson on the ranks. Captures full-vector ordering shifts that
    top-k Jaccard truncates and per-concept cosine ignores. Range
    ``[-1, 1]``; 1.0 = identical concept ordering.

    Args:
        clean_acts:   ``(B, K)``.
        corrupt_acts: ``(B, K)``.

    Returns:
        ``(B,)`` Spearman rho.
    """
    if clean_acts.shape != corrupt_acts.shape:
        raise ValueError(
            f"shape mismatch: {tuple(clean_acts.shape)} vs {tuple(corrupt_acts.shape)}"
        )
    rc = clean_acts.argsort(dim=-1).argsort(dim=-1).float()
    rp = corrupt_acts.argsort(dim=-1).argsort(dim=-1).float()
    rc = rc - rc.mean(dim=-1, keepdim=True)
    rp = rp - rp.mean(dim=-1, keepdim=True)
    num = (rc * rp).sum(dim=-1)
    denom = (rc.norm(dim=-1) * rp.norm(dim=-1)).clamp_min(1e-12)
    return num / denom
