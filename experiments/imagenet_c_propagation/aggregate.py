"""Stitch 50 per-cell artifacts into one dataset-level summary.

Run after the array job completes (via `--dependency=afterok`):

    python experiments/imagenet_c_propagation/aggregate.py

Reads:
    <HERE>/<corruption>_s<sev>/stage_metrics.jsonl  (50 files)
    <HERE>/<corruption>_s<sev>/per_image/<corruption>_s<sev>.pt  (50 files)

Writes:
    <HERE>/stage_metrics.jsonl    — concatenated rows from all 50 cells
    <HERE>/summary.json           — by-cell aggregated metrics +
                                    cross-severity prediction
                                    severity_shift_rate means per corruption family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbm_in_the_wild.corruption_eval import resolve_corruptions  # noqa: E402
from cbm_in_the_wild.dcbm.pipeline import compare_clean_vs_corrupted  # noqa: E402
from cbm_in_the_wild.runs import capture_context, write_summary  # noqa: E402


def _severity_shift_rate(seq: list[int]) -> float:
    """Adjacent-disagreement rate over a severity-indexed sequence.

    Continuum-axis primitive (severity 1→5) — kept local to keep the
    perturbation namespace clean of corruption-context callers.
    """
    if len(seq) < 2:
        return 0.0
    return sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b) / (len(seq) - 1)


def _concat_jsonls(corruptions: list[str], severities: tuple[int, ...]) -> Path:
    out = HERE / "stage_metrics.jsonl"
    if out.exists():
        out.unlink()
    missing = []
    n_rows = 0
    with open(out, "w") as wf:
        for c in corruptions:
            for s in severities:
                cell_jsonl = HERE / f"{c}_s{s}" / "stage_metrics.jsonl"
                if not cell_jsonl.exists():
                    missing.append(f"{c}_s{s}")
                    continue
                with open(cell_jsonl) as rf:
                    for line in rf:
                        wf.write(line)
                        n_rows += 1
    if missing:
        raise RuntimeError(f"missing per-cell JSONLs: {missing}")
    print(f"[aggregate] {n_rows} rows from {len(corruptions) * len(severities)} cells -> {out}")
    return out


def _severity_shift_rate_per_corruption(
    corruptions: list[str], severities: tuple[int, ...],
) -> dict[str, float]:
    """For each corruption: load 5 per_image .pt dumps, compute per-image
    prediction severity-shift-rate across the sev1→sev5 sequence, mean over images.
    """
    out: dict[str, float] = {}
    for c in corruptions:
        per_image_preds: list[torch.Tensor] = []
        for s in severities:
            pi_path = HERE / f"{c}_s{s}" / "per_image" / f"{c}_s{s}.pt"
            blob = torch.load(pi_path, map_location="cpu", weights_only=False)
            per_image_preds.append(blob["corrupt_pred"])
        stacked = torch.stack(per_image_preds, dim=0).t()
        n_images = stacked.shape[0]
        ssrs = [_severity_shift_rate([int(x) for x in stacked[i].tolist()]) for i in range(n_images)]
        out[c] = float(sum(ssrs) / n_images)
        print(f"[severity-shift-rate] {c}: mean={out[c]:.4f} n={n_images}")
    return out


def main() -> None:
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    corruptions = resolve_corruptions(cfg["corruption_set"])
    severities = tuple(cfg.get("severities", (1, 2, 3, 4, 5)))

    aggregated_jsonl = _concat_jsonls(corruptions, severities)
    by_cell = compare_clean_vs_corrupted(aggregated_jsonl)
    ssr_by_corruption = _severity_shift_rate_per_corruption(corruptions, severities)

    write_summary(
        HERE / "summary.json",
        config=cfg,
        metrics={
            "by_cell": by_cell,
            "n_cells": len(corruptions) * len(severities),
            "severity_shift_rate_mean_by_corruption": ssr_by_corruption,
        },
        timing_seconds={},
        context=capture_context(REPO_ROOT),
        extra={
            "corruptions": corruptions,
            "severities": list(severities),
        },
    )
    print(f"[summary] wrote {HERE / 'summary.json'}")


if __name__ == "__main__":
    main()
