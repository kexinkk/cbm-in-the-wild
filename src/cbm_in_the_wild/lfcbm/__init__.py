"""LF-CBM: training, propagation, d_k and capacity-matched ablation.

Wraps the forked upstream repo at ~/Label-free-CBM (never edited) the same
way dcbm_bridge wraps ~/DCBM. Deployed pipeline under test:

    CLIP ViT-B/16 z  →  W_c projection (concept scores)  →  per-concept
    standardize (train mean/std)  →  sparse GLM-SAGA head.

All corrupted inputs come from cached corrupted CLIP embeddings; no image
model ever runs on corrupted data here except the one-off CIFAR test-split
embedding pass (cells.py), which mirrors cifar10_c_propagation exactly.
"""

from .core import (  # noqa: F401
    LFCBM_REPO,
    load_artifacts,
    load_emb_split,
    logits_from_std,
    std_acts_from_emb,
)
