"""Adapter layer between our code and the upstream DCBM fork.

Every experiment that trains a DCBM should go through this module rather
than calling `utils.dcbm.CBM` directly. It encodes three things:

1. How to put `~/DCBM` (or $DCBM_REPO) on sys.path safely.
2. The seed-override discipline needed to defeat DCBM's hardcoded
   `torch.manual_seed(5)` at utils/dcbm.py:35.
3. How to recover final val/test metrics, which CBM.train() does not return.

The accuracy convention across our codebase is **fraction (0..1)**.
DCBM returns percent (0..100); we convert at the boundary here so
downstream code has one consistent scale.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cbm_in_the_wild.metrics_sink import MetricsSink
from cbm_in_the_wild.seeding import set_all_seeds


def _resolve_dcbm_repo(dcbm_repo: Path | None) -> Path:
    if dcbm_repo is not None:
        return Path(dcbm_repo)
    env = os.environ.get("DCBM_REPO")
    if env:
        return Path(env)
    return Path.home() / "DCBM"


def import_cbm(dcbm_repo: Path | None = None) -> type:
    """Add DCBM to sys.path and return the `CBM` class. Idempotent.

    The upstream module has a module-level `torch.manual_seed(5)` at line 35
    that fires on first import. Callers must re-seed afterwards — use
    `build_cbm` for that discipline instead of calling this directly.
    """
    repo = _resolve_dcbm_repo(dcbm_repo)
    if not repo.exists():
        raise FileNotFoundError(
            f"DCBM fork not found at {repo}. Set DCBM_REPO or pass dcbm_repo="
        )
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from utils.dcbm import CBM  # noqa: E402 — sys.path mutated above

    return CBM


def build_cbm(
    *,
    cfg: dict,
    seed: int,
    dcbm_repo: Path | None = None,
    metrics_sink: MetricsSink | None = None,
) -> Any:
    """Construct a `CBM` instance with our seed, not upstream's.

    Order of operations:
      1. set_all_seeds(seed)     — before upstream import fires
      2. import CBM              — triggers upstream torch.manual_seed(5)
      3. set_all_seeds(seed)     — overrides upstream
      4. cbm = CBM(...)          — construct
      5. set_all_seeds(seed)     — defensive: CBM.__init__ may touch RNGs

    After this returns, the RNGs are seeded to `seed`. Do NOT re-seed
    between this call and the next stochastic operation (clustering,
    training) or you will double-set.

    Keys read from `cfg`:
        dataset, model_name, class_labels_path (relative to DCBM/data/,
        or None), device.

    metrics_sink (optional, default None):
        If provided, attached to the returned CBM instance as
        ``cbm._metrics_sink`` for downstream runners to pick up. Default
        ``None`` keeps callers byte-identical — no attribute is set,
        no I/O happens. Opt-in.
    """
    set_all_seeds(seed)
    CBM = import_cbm(dcbm_repo)
    set_all_seeds(seed)

    repo = _resolve_dcbm_repo(dcbm_repo)
    embed_path = str(repo / "data" / "Embeddings") + "/"
    class_labels_rel = cfg.get("class_labels_path")
    class_labels = (
        str(repo / "data" / class_labels_rel) if class_labels_rel else None
    )

    cbm = CBM(
        embeddings_path=embed_path,
        dataset=cfg["dataset"],
        model=cfg["model_name"],
        class_labels_path=class_labels,
        device=cfg["device"],
    )
    set_all_seeds(seed)
    if metrics_sink is not None:
        cbm._metrics_sink = metrics_sink
    return cbm


def evaluate_split(
    cbm: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    batch_size: int,
    one_hot: bool = False,
) -> dict:
    """Run `cbm.evaluate` on (X, y) and return metrics as fractions.

    DCBM's `CBM.evaluate` returns `(avg_loss, accuracy_percent)` where
    accuracy is `100 * correct / total`. We divide by 100 once, here,
    so downstream code uses fractions consistently.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False
    )
    loss, acc_percent = cbm.evaluate(loader, device=device, one_hot=one_hot)
    return {"loss": float(loss), "accuracy": float(acc_percent) / 100.0}
