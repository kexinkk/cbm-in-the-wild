#!/usr/bin/env python
"""Aggregate the leave-one-corruption-out (LOCO) d_k selection runs.

For every family and held-out corruption c, compares accuracy on c (sev 3, 50k eval) of
  dk_loco_<c>  d_k from the other 9 corruptions (c unseen)
  dk_disjoint  d_k from all 10 corruptions (c seen), same disjoint pool
  rand0-2      size-matched random subsets
each refit under seeds 0-2. Writes loco/loco_summary.csv + loco/loco_summary.json.
Runs: run_loco.sh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent / "loco"
FAMILIES = ["fog", "gaussian_noise", "motion_blur", "patch_masking", "salt_and_pepper",
            "jpeg_compression", "row_column_dropout", "random_pixel_dropout",
            "sparse_random_occlusion", "structured_pixel_dropout"]


def _rows(model: str) -> pd.DataFrame:
    """Long table: rank, seed, corruption, acc."""
    recs = []
    if model == "dcbm":
        files = sorted((HERE / "dcbm").glob("*/capacity_matched/results.jsonl"))
        key, per = "arm", "corrupt_per_family"
    else:
        files = sorted((HERE / model).glob("*ablation_results.jsonl"))
        key, per = "rank", "corrupt_acc_per_family"
    for f in files:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            for c, acc in r[per].items():
                recs.append({"rank": r[key].removeprefix("B_prune_"), "seed": r["seed"],
                             "corruption": c, "acc": acc})
    return pd.DataFrame(recs)


def main() -> None:
    out, summ = [], {}
    for model in ["dcbm", "lfcbm", "pcbm_mmL"]:
        df = _rows(model)
        if df.empty:
            print(f"{model}: no results yet"); continue
        m = df.groupby(["rank", "corruption"])["acc"].mean().unstack() * 100   # rank x corruption
        rnd = m.loc[[r for r in m.index if r.startswith("rand")]]
        for c in FAMILIES:
            lo = f"dk_loco_{c}"
            if lo not in m.index or rnd.empty:
                continue
            out.append({
                "model": model, "held_out": c,
                "loco_acc": m.loc[lo, c], "seen_acc": m.loc["dk_disjoint", c] if "dk_disjoint" in m.index else np.nan,
                "random_acc": rnd[c].mean(),
                "loco_vs_random_pp": m.loc[lo, c] - rnd[c].mean(),
                "loco_vs_worst_random_pp": m.loc[lo, c] - rnd[c].max(),
                "seen_vs_random_pp": (m.loc["dk_disjoint", c] - rnd[c].mean()) if "dk_disjoint" in m.index else np.nan,
                "n_seeds_loco": int((df["rank"] == lo).sum() / len(FAMILIES)),
            })
        sub = pd.DataFrame([r for r in out if r["model"] == model])
        if not sub.empty:
            summ[model] = {
                "n_folds": len(sub),
                "mean_loco_vs_random_pp": float(sub["loco_vs_random_pp"].mean()),
                "min_loco_vs_random_pp": float(sub["loco_vs_random_pp"].min()),
                "folds_positive": int((sub["loco_vs_random_pp"] > 0).sum()),
                "mean_seen_vs_random_pp": float(sub["seen_vs_random_pp"].mean()),
            }
    res = pd.DataFrame(out)
    res.round(3).to_csv(HERE / "loco_summary.csv", index=False)
    (HERE / "loco_summary.json").write_text(json.dumps(summ, indent=2))
    print(res.round(2).to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
