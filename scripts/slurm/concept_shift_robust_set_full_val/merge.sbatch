#!/bin/bash
# Merge all full-val shards into the canonical artifacts. Pure I/O + pandas groupby (no
# embeddings, no refit): concatenate every shards/*/capacity_matched/results.jsonl, dedup by
# (arm,K,seed,fit_n), and write the per-cell CSV + summary the notebook reads. Runs afterany on
# the shard jobs (a failed shard just leaves its cells missing — visible as gaps in the CSV).

#SBATCH --job-name=cmfv-merge
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) @ $(date -Iseconds)"
source /usr/share/lmod/lmod/init/profile 2>/dev/null || true
module purge
module load devel/miniforge/25.3.1-python-3.12
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"
cd "${PROJECT_ROOT}"

srun python - <<'PY'
import json, pathlib, pandas as pd
FV = pathlib.Path("experiments/imagenet_concept_shift_robust_set_full_val")
shard_files = sorted(FV.glob("shards/*/capacity_matched/results.jsonl"))
rows = {}
for f in shard_files:
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[(r["arm"], r["K_wanted"], r["seed"], r.get("fit_n"))] = r   # dedup, keep last
recs = list(rows.values())
print(f"merged {len(recs)} cells from {len(shard_files)} shards")

# canonical per-cell results.jsonl (what a normal run would have produced)
out_cache = FV / "capacity_matched"; out_cache.mkdir(parents=True, exist_ok=True)
with (out_cache / "results.jsonl").open("w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")

# per-cell CSV (notebook does its own groupby); sorted for stable diffs
ab = pd.DataFrame(recs).sort_values(["arm", "K_wanted", "seed"]).reset_index(drop=True)
ab.to_csv(FV / "ablation_capacity_matched.csv", index=False)

fit_ns = sorted(ab["fit_n"].dropna().unique().tolist()) if "fit_n" in ab else []
# max_iter is the lbfgs BUDGET (for the notebook convergence audit) — read from a shard summary,
# NOT the observed max n_iter. Fallback 1000 (the runner default).
_ss = sorted(FV.glob("shards/*/ablation_capacity_matched_summary.json"))
max_iter = 1000
for s in _ss:
    try:
        max_iter = int(json.loads(s.read_text())["max_iter"]); break
    except Exception:
        pass
summary = {
    "method": "capacity_matched_A_native_vs_B_prune", "source": "sharded_full_val_merge",
    "fit_n": int(fit_ns[-1]) if fit_ns else None,
    "max_iter": max_iter,
    "sizes": sorted(int(k) for k in ab["K_wanted"].unique()),
    "seeds": sorted(int(s) for s in ab["seed"].unique()),
    "arms": sorted(ab["arm"].unique().tolist()),
    "n_cells": len(ab), "n_shards": len(shard_files),
    "max_n_iter_observed": int(ab["n_iter"].max()) if "n_iter" in ab and len(ab) else None,
}
(FV / "ablation_capacity_matched_summary.json").write_text(json.dumps(summary, indent=2))
print("wrote CSV + summary. arms:", summary["arms"])
print("cells per arm:\n" + ab.groupby("arm").size().to_string())
# flag an incomplete grid (missing shard cells)
expect = len(summary["sizes"]) * len(summary["seeds"])
short = {a: n for a, n in ab.groupby("arm").size().items() if n < expect}
if short:
    print(f"WARNING incomplete arms (<{expect} cells): {short}")
PY
echo "Finished @ $(date -Iseconds) -> experiments/imagenet_concept_shift_robust_set_full_val/ablation_capacity_matched.csv"
