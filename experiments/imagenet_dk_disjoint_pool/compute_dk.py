#!/usr/bin/env python
"""Per-concept shift d_k on an image pool disjoint from the evaluation images.

The paper's d_k (imagenet_concept_shift_robust_set, lfcbm/pcbm concept_shift.py)
is estimated on the same 50k ImageNet val images the selection arms are evaluated
on. This re-estimates it on the 50/class ImageNet *train* fit pool (50,000 images,
disjoint from the 50k eval set and from the 128,118-image refit pool), using the
seeded corrupted-train CLIP cells (embed_train_pool.py, PYTHONHASHSEED=0) and the clean
DCBM train embeddings. Score spaces are unchanged per family:
    DCBM      z·c_k/‖c_k‖²      (imagenet_baseline centroids)
    LF-CBM    W_c z             (imagenet_lfcbm_genericity/model)
    PCBM-mmL  z·c_k/‖c_k‖²      (imagenet_pcbm_genericity/model_mmL)

Writes <family>_concept_shift_table.csv (d_k, d_k_hold3) + <family>_shift_grids.npz
+ summary.json (Spearman vs the eval-pool ranking). The ablation runners read
d_k (all severities) via env DK_DISJOINT_TABLE (rank token dk_disjoint).

"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from cbm_in_the_wild.paths import CHECKPOINT_DIR, DATA_DIR  # noqa: E402
from cbm_in_the_wild.persistence import load_cbm_artifacts  # noqa: E402
from cbm_in_the_wild.lfcbm import core as lf_core  # noqa: E402
from cbm_in_the_wild.pcbm import core as pc_core  # noqa: E402

# corrupted-train CLIP cells written by embed_train_pool.py
CELL_ROOT = Path(os.environ.get("DK_POOL_CELL_ROOT", DATA_DIR / "dk_disjoint_pool" / "train_cells"))
FAMILIES = ["fog", "gaussian_noise", "motion_blur", "patch_masking", "salt_and_pepper",
            "jpeg_compression", "row_column_dropout", "random_pixel_dropout",
            "sparse_random_occlusion", "structured_pixel_dropout"]
SEVERITIES = [1, 2, 3, 4, 5]
# Convention (author, 2026-10-13): d_k averages all five severities for every family,
# no severity-3 hold-out. d_k_hold3 is still written, for the record only.
EVAL_POOL_TABLES = {   # eval-pool d_k (all severities), for the Spearman comparison
    "dcbm": REPO / "experiments/imagenet_concept_shift_robust_set/concept_shift_table.csv",
    "lfcbm": REPO / "experiments/imagenet_lfcbm_genericity_full_val/concept_shift_table.csv",
    "pcbm_mmL": REPO / "experiments/imagenet_pcbm_genericity_full_val/mmL_concept_shift_table.csv",
}


def score_fns() -> dict:
    art = load_cbm_artifacts(CHECKPOINT_DIR / "imagenet_baseline")
    C = art["centroids"].float()
    nsq = (C ** 2).sum(1)
    lf = lf_core.load_artifacts(REPO / "experiments/imagenet_lfcbm_genericity_full_val/model")
    pc = pc_core.load_artifacts(REPO / "experiments/imagenet_pcbm_genericity_full_val/model_mmL")
    return {
        "dcbm": lambda z: (z @ C.t()) / nsq,
        "lfcbm": lambda z: lf_core.raw_acts_from_emb(z, lf),
        "pcbm_mmL": lambda z: pc_core.raw_acts_from_emb(z, pc),
    }


def eval_pool_dk(fam: str) -> np.ndarray:
    df = pd.read_csv(EVAL_POOL_TABLES[fam])
    return df.sort_values("concept_id")["d_k"].values


def main() -> None:
    torch.set_grad_enabled(False)
    fns = score_fns()

    first = torch.load(CELL_ROOT / f"{FAMILIES[0]}_sev1.pt", map_location="cpu")
    pool_ids = list(first["ids"])
    ids, X, _, _ = lf_core.load_emb_split("imagenet", "train")
    row = {i: k for k, i in enumerate(ids)}
    clean_z = X[torch.tensor([row[i] for i in pool_ids])].float()
    print(f"pool {len(pool_ids)} images, clean z {tuple(clean_z.shape)}", flush=True)
    clean = {f: fn(clean_z) for f, fn in fns.items()}

    grids = {f: np.full((c.shape[1], len(FAMILIES), len(SEVERITIES)), np.nan)
             for f, c in clean.items()}
    cos_log = {}
    for fi, corr in enumerate(FAMILIES):
        for si, sev in enumerate(SEVERITIES):
            d = torch.load(CELL_ROOT / f"{corr}_sev{sev}.pt", map_location="cpu")
            assert list(d["ids"]) == pool_ids, f"id order mismatch in {corr}_sev{sev}"
            z = d["clip_emb"].float()
            cos_log[f"{corr}_s{sev}"] = float(torch.nn.functional.cosine_similarity(z, clean_z).mean())
            for f, fn in fns.items():
                grids[f][:, fi, si] = (fn(z) - clean[f]).abs().mean(0).numpy()
        print(f"done {corr}: cos s1..s5 "
              + " ".join(f"{cos_log[f'{corr}_s{s}']:.3f}" for s in SEVERITIES), flush=True)

    hold = [i for i, s in enumerate(SEVERITIES) if s != 3]
    summary = {"pool": "imagenet train fit pool (50/class), disjoint from 50k eval + 128k refit",
               "n_images": len(pool_ids), "cell_root": str(CELL_ROOT),
               "mean_cos_clean_vs_corrupt": cos_log, "families": {}}
    for f, g in grids.items():
        dk, dk_h3 = np.nanmean(g, axis=(1, 2)), np.nanmean(g[:, :, hold], axis=(1, 2))
        np.savez(HERE / f"{f}_shift_grids.npz", dk_grid=g,
                 families=np.array(FAMILIES), severities=np.array(SEVERITIES))
        pd.DataFrame({"concept_id": np.arange(len(dk)), "d_k": dk, "d_k_hold3": dk_h3}) \
          .to_csv(HERE / f"{f}_concept_shift_table.csv", index=False)
        ev = eval_pool_dk(f)
        n = len(dk)
        top = lambda v, q: set(np.argsort(v)[: int(q * n)])
        summary["families"][f] = {
            "n_concepts": n,
            "spearman_disjoint_vs_eval": float(spearmanr(dk, ev).statistic),
            "robust20pct_overlap": len(top(dk, .2) & top(ev, .2)) / int(.2 * n),
            "dk_mean": float(dk.mean()), "eval_dk_mean": float(ev.mean()),
        }
        print(f, summary["families"][f], flush=True)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
