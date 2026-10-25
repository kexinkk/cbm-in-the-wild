"""PCBM: per-concept corruption shift d_k.

Copied from lfcbm/concept_shift.py (bank-agnostic; operates only on the raw
[N,K] concept-score matrix). d_k = mean absolute per-concept shift between
clean and corrupted scores, averaged over images then over (family, severity),
on the *raw* PCBM margins m_k = (z·cav_k + b_k)/‖cav_k‖ (pre-standardization) —
the DCBM/LF-CBM-comparable raw-score space. Emits d_k^hold3 (severity 3
excluded → no leakage into the sev-3 ablation eval) and, for the record, the
standardized twin d_k_std = d_k / σ_k (σ_k = proj_std).

Runs per concept source: cfg["bank"] ∈ {cav, mm} selects model_<bank>/ and
prefixes all outputs with <bank>_ so the two banks never collide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .cells import load_cell
from .core import load_artifacts, load_emb_split, raw_acts_from_emb


def run(cfg: dict, exp_dir: Path) -> None:
    bank = cfg.get("bank", "cav")
    model_dir = exp_dir / f"model_{bank}"

    def _out(name: str) -> Path:
        return exp_dir / f"{bank}_{name}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = load_artifacts(model_dir, device=device)
    ids, X, _, _ = load_emb_split(cfg["dataset"], "test")
    row_of = {i: k for k, i in enumerate(ids)}
    with torch.no_grad():
        clean_acts = raw_acts_from_emb(X.to(device), art)

    from cbm_in_the_wild.corruption_eval import resolve_corruptions

    families = resolve_corruptions(cfg["corruption_set"])
    severities = list(cfg["severities"])
    K = clean_acts.shape[1]
    dk_grid = np.full((K, len(families), len(severities)), np.nan, dtype=np.float32)

    for fi, fam in enumerate(families):
        for si, sev in enumerate(severities):
            cell = load_cell(cfg, exp_dir, fam, sev)
            idx = torch.tensor([row_of[i] for i in cell["image_ids"]],
                               dtype=torch.long, device=device)
            with torch.no_grad():
                a_co = raw_acts_from_emb(cell["clip_emb"].to(device), art)
                d = (a_co - clean_acts[idx]).abs().mean(0)
            dk_grid[:, fi, si] = d.cpu().numpy()
            print(f"[dk:{bank}] {fam}_s{sev} mean {float(d.mean()):.4f}")

    dk = np.nanmean(dk_grid, axis=(1, 2))
    hold = [si for si, s in enumerate(severities) if s != 3]
    dk_hold3 = np.nanmean(dk_grid[:, :, hold], axis=(1, 2))
    w_imp = art["W_g"].abs().max(0).values.cpu().numpy()

    sigma = art["proj_std"].flatten().cpu().numpy()
    dk_std = dk / sigma
    dk_std_hold3 = dk_hold3 / sigma

    np.savez(_out("shift_grids.npz"), dk_grid=dk_grid, sigma=sigma,
             families=np.array(families), severities=np.array(severities))

    import pandas as pd

    df = pd.DataFrame({
        "concept_id": np.arange(K),
        "name": art["concepts"],
        "d_k": dk,
        "d_k_hold3": dk_hold3,
        "d_k_std": dk_std,
        "d_k_std_hold3": dk_std_hold3,
        "w_imp": w_imp,
    })
    df.to_csv(_out("concept_shift_table.csv"), index=False)

    from scipy.stats import spearmanr

    summary = {
        "bank": bank,
        "n_concepts": int(K),
        "dk_space": "raw_pcbm_projection (⟨z,c_k⟩/‖c_k‖², PCBM paper §2 form — no intercept, "
                    "squared norm; pre-standardization; DCBM-comparable)",
        "dk_mean": float(dk.mean()),
        "dk_p90_over_p10": float(np.percentile(dk, 90) / np.percentile(dk, 10)),
        "spearman_dk_vs_hold3": float(spearmanr(dk, dk_hold3).statistic),
        "spearman_dk_vs_wimp": float(spearmanr(dk, w_imp).statistic),
        "spearman_dk_raw_vs_std": float(spearmanr(dk, dk_std).statistic),
    }
    (_out("concept_shift_summary.json")).write_text(json.dumps(summary, indent=2))
    print(f"[dk:{bank}] summary", summary)
