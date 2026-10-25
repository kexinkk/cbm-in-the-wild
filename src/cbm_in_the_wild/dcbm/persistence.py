"""Persist + restore the post-training DCBM artifacts.

Upstream ``CBM.train()`` keeps the best linear-head weights in memory only;
``cluster_image_concepts`` / ``centroid_concepts`` only persist when
``to_save_concepts=True`` (default ``False``). Neither is currently saved by
our baseline runners, so the trained head + concept bank disappear with
each SLURM job.

These artifacts are used across the project — Phase-2 corruption eval,
Phase-3 adaptation, the stage-logging smoke. This module gives all of
them one persistence contract.

Layout under ``out_dir``::

    linear_head.pt       torch.save(cbm.model.state_dict())  → keys "linear.weight" / "linear.bias"
    centroids.pt         torch.save(Tensor (K, D))           → CLIP-D=512 for ViT-B/16
    scaler.pkl           pickle.dump(cbm.scaler)             → sklearn StandardScaler
    manifest.json        dataset, n_clusters, num_classes, clip_dim, lambda_1, lr, seed, ...

``load_cbm_artifacts`` returns a dict of loaded objects + a ready-to-use
``LinearProbe`` instance (the trivial 1-linear-layer wrapper upstream uses).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch import nn


_LINEAR_HEAD = "linear_head.pt"
_CENTROIDS = "centroids.pt"
_SCALER = "scaler.pkl"
_MANIFEST = "manifest.json"


class LinearProbe(nn.Module):
    """Mirror of upstream's nested ``LinearProbe`` (utils/dcbm.py:945).

    Replicated here so loaded ``state_dict`` keys (``linear.weight`` /
    ``linear.bias``) match without needing to import upstream.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def save_cbm_artifacts(cbm: Any, out_dir: Path, *, manifest: dict | None = None) -> Path:
    """Save linear head + centroids + scaler + manifest.

    ``cbm`` must be post-train (``cbm.model``, ``cbm.scaler``,
    ``cbm.clustered_concepts`` populated). ``manifest`` is merged into the
    auto-derived fields (dataset, shapes); caller-supplied values win.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(cbm.model.state_dict(), out / _LINEAR_HEAD)
    centroids = torch.as_tensor(cbm.clustered_concepts, dtype=torch.float32)
    torch.save(centroids, out / _CENTROIDS)
    with open(out / _SCALER, "wb") as f:
        pickle.dump(cbm.scaler, f)

    # Shape facts are derived from the saved tensors and override any caller
    # input — upstream KMeans silently caps n_clusters at the available
    # concept-crop count (e.g. cifar10 request=1024 → got=215), so the
    # configured value is not trustworthy.
    facts = {
        "dataset": getattr(cbm, "dataset", None),
        "n_clusters": int(centroids.shape[0]),
        "clip_dim": int(centroids.shape[1]),
        "num_classes": int(cbm.model.linear.weight.shape[0]),
    }
    merged = {**(manifest or {}), **facts}
    with open(out / _MANIFEST, "w") as f:
        json.dump(merged, f, indent=2)
    return out


def load_cbm_artifacts(in_dir: Path, *, device: str | torch.device = "cpu") -> dict:
    """Load everything ``save_cbm_artifacts`` wrote.

    Returns::

        {
          "linear_head": LinearProbe (loaded, on `device`, eval mode),
          "centroids":   Tensor (K, D) on cpu,
          "scaler":      sklearn.preprocessing.StandardScaler,
          "manifest":    dict,
        }
    """
    src = Path(in_dir)
    with open(src / _MANIFEST) as f:
        manifest = json.load(f)
    centroids = torch.load(src / _CENTROIDS, map_location="cpu")
    with open(src / _SCALER, "rb") as f:
        scaler = pickle.load(f)

    head_state = torch.load(src / _LINEAR_HEAD, map_location="cpu")
    # Derive shapes from the actual tensors — manifest can drift (see
    # save_cbm_artifacts).
    num_classes, input_dim = head_state["linear.weight"].shape
    head = LinearProbe(int(input_dim), int(num_classes))
    head.load_state_dict(head_state)
    head.to(device).eval()

    return {
        "linear_head": head,
        "centroids": centroids,
        "scaler": scaler,
        "manifest": manifest,
    }
