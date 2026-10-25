"""Pure-function stage-wise metrics.

Each submodule scores one pipeline stage. No I/O, no DCBM / SAM2 / CLIP
imports — callers pass pre-computed tensors in, these functions produce
numbers out. Wiring into an evaluation pipeline is intentionally out of
scope for these modules.

Modules:
    - ``clip_drift``     — cosine-based drift on vision-encoder embeddings.
    - ``concept_stats``  — entropy / cosine / top-k Jaccard on concept activations.
    - ``adaptation``     — linear-layer weight shift between two heads.
    - ``prediction``     — accuracy, mCE, RmCE (Hendrycks and Dietterich, 2019).
    - ``perturbation``   — flip-rate / mFR for video-like perturbation sequences.
"""

from cbm_in_the_wild.metrics.adaptation import linear_weight_shift
from cbm_in_the_wild.metrics.clip_drift import (
    cosine_drift,
    cosine_drift_matrix,
    nearest_ref_drift,
)
from cbm_in_the_wild.metrics.concept_stats import (
    argmax_shift_fraction,
    entropy_shift,
    kl_divergence,
    per_concept_cosine,
    softmax_entropy,
    spearman_rho,
    top_k_concept_overlap,
)
from cbm_in_the_wild.metrics.perturbation import (
    flip_rate,
    mfr_predictions,
    mfr_ratio,
    stage_wise_flip_rate,
)
from cbm_in_the_wild.metrics.prediction import accuracy, mce, rmce

__all__ = [
    "accuracy",
    "argmax_shift_fraction",
    "cosine_drift",
    "cosine_drift_matrix",
    "entropy_shift",
    "flip_rate",
    "kl_divergence",
    "linear_weight_shift",
    "mce",
    "mfr_predictions",
    "mfr_ratio",
    "nearest_ref_drift",
    "per_concept_cosine",
    "rmce",
    "softmax_entropy",
    "spearman_rho",
    "stage_wise_flip_rate",
    "top_k_concept_overlap",
]
