#!/usr/bin/env python
"""Aggregate the leave-one-corruption-out (LOCO) arms of the family-recipe retrain.

For every family and held-out corruption c, at the family's peak-margin K, compares
severity-3 accuracy on c (50k test, mean over seeds 0-2) of
  loco_<c>_<K>  disjoint-pool d_k from the other 9 corruptions (c unseen)
  dkd<K>        disjoint-pool d_k from all 10 corruptions (c seen)
  rand<K>       size-matched random subsets
Retention = mean(loco - rand) / mean(dkd - rand). Writes loco_retrain.csv +
loco_retrain.json. Runs: ../imagenet_pruned_bank_deployed_retrain (DCBM) and
../imagenet_pruned_concepts_family_retrain (LF-CBM, PCBM-mmL).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
K = {"dcbm": 200, "lfcbm": 400, "pcbm_mmL": 3600}
CORRUPTIONS = ["fog", "gaussian_noise", "motion_blur", "patch_masking", "salt_and_pepper",
               "jpeg_compression", "row_column_dropout", "random_pixel_dropout",
               "sparse_random_occlusion", "structured_pixel_dropout"]


def _rows() -> pd.DataFrame:
    """Long table: family, arm, seed, corruption, acc (severity 3)."""
    recs = []
    for line in (EXP / "imagenet_pruned_bank_deployed_retrain/eval_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs += [{"family": "dcbm", "arm": r["arm"], "seed": r["seed"], "corruption": c, "acc": a}
                 for c, a in r["per_family"].items()]
    for f in sorted((EXP / "imagenet_pruned_concepts_family_retrain/runs").glob("*/*.json")):
        r = json.loads(f.read_text())
        recs += [{"family": r["family"], "arm": r["arm"], "seed": r["seed"], "corruption": c,
                  "acc": r["cells"][f"{c}_s3"]} for c in CORRUPTIONS]
    return pd.DataFrame(recs)


def main() -> None:
    m = _rows().groupby(["family", "arm", "corruption"])["acc"].mean() * 100
    out, summ = [], {}
    for fam, k in K.items():
        for c in CORRUPTIONS:
            loco, dkd, rnd = (m[fam, f"loco_{c}_{k}", c], m[fam, f"dkd{k}", c], m[fam, f"rand{k}", c])
            out.append({"family": fam, "K": k, "held_out": c, "loco": loco, "dkd": dkd, "rand": rnd,
                        "loco_vs_rand": loco - rnd, "loco_vs_dkd": loco - dkd})
        sub = pd.DataFrame(out[-len(CORRUPTIONS):])
        seen = (sub["dkd"] - sub["rand"]).mean()
        summ[fam] = {"K": k,
                     "mean_loco_vs_rand_pp": float(sub["loco_vs_rand"].mean()),
                     "mean_dkd_vs_rand_pp": float(seen),
                     "retention": float(sub["loco_vs_rand"].mean() / seen),
                     "folds_loco_above_rand": int((sub["loco_vs_rand"] > 0).sum())}
    res = pd.DataFrame(out)
    res.to_csv(HERE / "loco_retrain.csv", index=False)
    (HERE / "loco_retrain.json").write_text(json.dumps(summ, indent=2))
    print(res.round(2).to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
