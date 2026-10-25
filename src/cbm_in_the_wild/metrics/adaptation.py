"""Linear-layer adaptation metrics: weight shift between two heads.

Scores how much a linear prediction layer's weight matrix moved between
two training conditions (e.g. clean-trained vs fine-tuned-on-corruption).
Pure-function: pass in two ``(C, K)`` weight matrices, get back per-row
shift statistics + a global Frobenius norm.
"""

from __future__ import annotations

import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def linear_weight_shift(W_clean: torch.Tensor, W_adapted: torch.Tensor) -> dict:
    """Per-row + global shift between two linear-layer weight matrices.

    For a head with weight matrix ``(C, K)`` (one row per output class,
    weighted over K input features), reports:

    - **per-row cosine** between W_clean[i] and W_adapted[i]: how class i's
      input-weighting rotated. 1.0 = identical, 0 = orthogonal.
    - **per-row L2 magnitude shift**: ``||W_adapted[i]|| - ||W_clean[i]||``.
    - **global Frobenius distance**: ``||W_adapted - W_clean||_F``.
    - **relative Frobenius**: Frobenius distance / ``||W_clean||_F`` —
      scale-free comparison across class counts.

    Args:
        W_clean:   ``(C, K)`` weight matrix of the reference head.
        W_adapted: ``(C, K)`` weight matrix of the compared head.

    Returns:
        ``{"per_class_cosine": (C,), "per_class_magnitude_delta": (C,),
          "frobenius": float, "relative_frobenius": float}``.

    Raises:
        ValueError: shape mismatch, or W_clean has zero Frobenius norm.
    """
    if W_clean.shape != W_adapted.shape:
        raise ValueError(
            f"shape mismatch: {tuple(W_clean.shape)} vs {tuple(W_adapted.shape)}"
        )
    if W_clean.ndim != 2:
        raise ValueError(f"expected (C, K) 2D matrices, got {W_clean.ndim}D")

    cos = (_l2_normalize(W_clean) * _l2_normalize(W_adapted)).sum(dim=-1)
    mag_delta = W_adapted.norm(dim=-1) - W_clean.norm(dim=-1)
    fro = (W_adapted - W_clean).norm().item()
    fro_clean = W_clean.norm().item()
    if fro_clean == 0.0:
        raise ValueError("W_clean has zero Frobenius norm — relative shift undefined")

    return {
        "per_class_cosine": cos,
        "per_class_magnitude_delta": mag_delta,
        "frobenius": float(fro),
        "relative_frobenius": float(fro) / float(fro_clean),
    }
