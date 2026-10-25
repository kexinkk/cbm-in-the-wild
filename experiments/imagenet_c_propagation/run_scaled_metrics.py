"""Recompute stage-2 metrics on StandardScaler-transformed concept activations.

Uses the same cell infrastructure as run_cell.py but only emits the three
rank-based metrics (argmax_shift, spearman_rho, top5_jaccard) computed on
the scaled activations (what the linear head actually reads), not the raw
post-softmax vector.

Writes `scaled_stage_metrics.jsonl` per cell and a merged
`scaled_stage_metrics.jsonl` at the experiment root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbm_in_the_wild.corruption_eval import (
    resolve_corruptions,
    setup_for_cells,
)
from cbm_in_the_wild.metrics import (
    argmax_shift_fraction,
    spearman_rho,
    top_k_concept_overlap,
)
from cbm_in_the_wild.dcbm.pipeline import propagate_corrupted_cell, MetricsSink


def run_scaled_cell(ctx, corruption, severity, cell_dir):
    """Re-forward one cell and compute metrics on scaled activations."""
    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    scaler = ctx.artifacts["scaler"]

    t0 = time.time()
    clip_emb_chunks = []
    batch_size = 512
    for start in range(0, len(ctx.eval_ids), batch_size):
        chunk_ids = ctx.eval_ids[start : start + batch_size]
        chunk_imgs = ctx.corrupt_fn(chunk_ids, corruption, severity)
        chunk_emb = ctx.clip_forward(chunk_imgs)
        clip_emb_chunks.append(chunk_emb)
        del chunk_imgs
    clip_emb = torch.cat(clip_emb_chunks, dim=0)
    del clip_emb_chunks

    centroids = ctx.artifacts["centroids"].to(ctx.device)
    norms_sq_t = (centroids ** 2).sum(dim=1, keepdim=True).t()
    raw = (clip_emb.to(ctx.device).float() @ centroids.t()) / norms_sq_t
    corrupt_acts = torch.nn.functional.softmax(raw, dim=1).cpu()

    clean_acts = ctx.clean_eval["concept_acts"]

    scaled_clean = torch.tensor(
        scaler.transform(clean_acts.numpy()), dtype=torch.float32
    )
    scaled_corrupt = torch.tensor(
        scaler.transform(corrupt_acts.numpy()), dtype=torch.float32
    )

    asf = argmax_shift_fraction(scaled_clean, scaled_corrupt)
    sp = spearman_rho(scaled_clean, scaled_corrupt)
    tk = top_k_concept_overlap(scaled_clean, scaled_corrupt, k=5)

    wall = time.time() - t0

    row = {
        "dataset": ctx.dataset,
        "corruption": corruption,
        "severity": severity,
        "scaled_argmax_shift_fraction": asf,
        "scaled_spearman_rho_mean": float(sp.mean()),
        "scaled_top5_jaccard_mean": float(tk.mean()),
        "wall_seconds": round(wall, 1),
    }

    out_path = cell_dir / "scaled_stage_metrics.jsonl"
    with open(out_path, "w") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[scaled] {corruption} sev={severity} done in {wall:.1f}s -> {out_path}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="cell index [0,49]; defaults to SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="run all 50 cells sequentially (for single-job mode)",
    )
    args = parser.parse_args()

    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    corruptions = resolve_corruptions(cfg["corruption_set"])
    severities = tuple(cfg.get("severities", (1, 2, 3, 4, 5)))
    n_cells = len(corruptions) * len(severities)

    ctx = setup_for_cells(cfg, full_test=bool(cfg.get("full_test", True)))

    if args.all:
        rows = []
        for task_id in range(n_cells):
            corruption = corruptions[task_id // len(severities)]
            severity = severities[task_id % len(severities)]
            cell_dir = HERE / f"{corruption}_s{severity}"
            row = run_scaled_cell(ctx, corruption, severity, cell_dir)
            rows.append(row)
        merged = HERE / "scaled_stage_metrics.jsonl"
        with open(merged, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[merged] wrote {len(rows)} rows -> {merged}")
    else:
        task_id = args.task_id
        if task_id is None:
            env = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env is None:
                raise RuntimeError("pass --task-id or --all, or run under SLURM array")
            task_id = int(env)
        if not 0 <= task_id < n_cells:
            raise ValueError(f"task-id {task_id} out of range [0, {n_cells})")
        corruption = corruptions[task_id // len(severities)]
        severity = severities[task_id % len(severities)]
        cell_dir = HERE / f"{corruption}_s{severity}"
        run_scaled_cell(ctx, corruption, severity, cell_dir)


if __name__ == "__main__":
    main()
