"""LF-CBM: per-concept corruption shift d_k.

d_k mirrors experiments/imagenet_concept_shift_robust_set/run.py exactly:
mean absolute per-concept shift between clean and corrupted scores, averaged
over images then over (family, severity) cells, on the *raw* concept
projection scores c_k = W_c z (pre-standardization). This matches DCBM's
raw pre-softmax d_k definition — W_c rows are near-unit-norm, so raw scores
are already cross-concept comparable (verified 2026-07-28: σ_k and ‖W_c,k‖
both span only ~2x). The head still reads standardized activations; only
the *ranking metric* lives in raw space (as in DCBM, whose ablation ranks
on raw d_k but runs the true softmax→scaler→head forward path).

Also emits d_k^hold3 (severity 3 excluded) — the no-leakage twin used by
the capacity-matched ablation, whose eval severity is 3 — and, for the
record, the previously-used standardized twins d_k_std(_hold3) = d_k / σ_k
(σ_k = proj_std). Because standardization only rescales by the per-concept
σ_k, d_k_std = d_k / σ_k exactly; the two rankings nonetheless diverge in
the low-d_k tail the selection test uses (ImageNet keep-lowest-50 Jaccard
≈ 0.10), so the ablation carries a std reference arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .cells import load_cell
from .core import load_artifacts, load_emb_split, raw_acts_from_emb


def run(cfg: dict, exp_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = load_artifacts(exp_dir / "model", device=device)
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
            print(f"[dk] {fam}_s{sev} mean {float(d.mean()):.4f}")

    dk = np.nanmean(dk_grid, axis=(1, 2))                        # raw projection shift
    hold = [si for si, s in enumerate(severities) if s != 3]
    dk_hold3 = np.nanmean(dk_grid[:, :, hold], axis=(1, 2))
    w_imp = art["W_g"].abs().max(0).values.cpu().numpy()

    # recorded reference: the previously-used standardized twin. Standardizing
    # only rescales by σ_k, so d_k_std = d_k / σ_k exactly (see module docstring).
    sigma = art["proj_std"].flatten().cpu().numpy()
    dk_std = dk / sigma
    dk_std_hold3 = dk_hold3 / sigma

    np.savez(exp_dir / "shift_grids.npz", dk_grid=dk_grid, sigma=sigma,
             families=np.array(families), severities=np.array(severities))

    import pandas as pd

    df = pd.DataFrame({
        "concept_id": np.arange(K),
        "name": art["concepts"],
        "d_k": dk,
        "d_k_hold3": dk_hold3,
        "d_k_std": dk_std,              # recorded reference (old definition)
        "d_k_std_hold3": dk_std_hold3,
        "w_imp": w_imp,
    })
    df.to_csv(exp_dir / "concept_shift_table.csv", index=False)

    from scipy.stats import spearmanr

    rho = float(spearmanr(dk, dk_hold3).statistic)
    summary = {
        "n_concepts": int(K),
        "dk_space": "raw_projection (W_c z, pre-standardization; DCBM-comparable)",
        "dk_mean": float(dk.mean()),
        "dk_p90_over_p10": float(np.percentile(dk, 90) / np.percentile(dk, 10)),
        "spearman_dk_vs_hold3": rho,
        "spearman_dk_vs_wimp": float(spearmanr(dk, w_imp).statistic),
        "spearman_dk_raw_vs_std": float(spearmanr(dk, dk_std).statistic),
        "spearman_dk_raw_vs_std_hold3": float(spearmanr(dk_hold3, dk_std_hold3).statistic),
    }
    (exp_dir / "concept_shift_summary.json").write_text(json.dumps(summary, indent=2))
    print("[dk] summary", summary)
