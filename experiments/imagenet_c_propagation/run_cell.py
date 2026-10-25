"""ImageNet Tier-2 array driver — one (corruption, severity) cell per task.

Reads `$SLURM_ARRAY_TASK_ID` (or `--task-id` for local debug) and maps to
`(corruptions[task_id // 5], severities[task_id % 5])` using the same
deterministic order as `corruption_eval.resolve_corruptions("combined")`:

    task 0..4   -> motion_blur s1..s5
    task 5..9   -> gaussian_noise s1..s5
    task 10..14 -> jpeg_compression s1..s5
    task 15..19 -> fog s1..s5
    task 20..24 -> random_pixel_dropout s1..s5
    task 25..29 -> salt_and_pepper s1..s5
    task 30..34 -> sparse_random_occlusion s1..s5
    task 35..39 -> patch_masking s1..s5
    task 40..44 -> row_column_dropout s1..s5
    task 45..49 -> structured_pixel_dropout s1..s5

Per-cell artifacts at `experiments/imagenet_c_propagation/<corruption>_s<sev>/`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbm_in_the_wild.corruption_eval import (  # noqa: E402
    resolve_corruptions,
    run_one_cell,
    setup_for_cells,
)


def _resolve_task_id(arg: int | None) -> int:
    if arg is not None:
        return int(arg)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise RuntimeError(
            "no task id: pass --task-id or run under SLURM with --array=..."
        )
    return int(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="cell index in [0, 49]; defaults to SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument(
        "--dump-stage-tensors", action="store_true",
        help="persist per-image clip_emb (N,512) + concept_acts (N,K) to "
             "per_image/<corruption>_s<sev>.pt; opt-in for selected cells "
             "(see PROPOSAL_visualization_0505.md §5).",
    )
    args = parser.parse_args()
    task_id = _resolve_task_id(args.task_id)

    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    corruptions = resolve_corruptions(cfg["corruption_set"])
    severities = tuple(cfg.get("severities", (1, 2, 3, 4, 5)))
    n_cells = len(corruptions) * len(severities)
    if not 0 <= task_id < n_cells:
        raise ValueError(f"task-id {task_id} out of range [0, {n_cells})")

    corruption = corruptions[task_id // len(severities)]
    severity = severities[task_id % len(severities)]
    cell_dir = HERE / f"{corruption}_s{severity}"
    print(f"[task {task_id}] corruption={corruption} severity={severity} -> {cell_dir}")

    full_test = bool(cfg.get("full_test", True))
    ctx = setup_for_cells(cfg, full_test=full_test)
    run_one_cell(
        ctx, corruption, severity,
        cell_dir=cell_dir, cfg_for_summary=cfg,
        dump_stage_tensors=args.dump_stage_tensors,
    )


if __name__ == "__main__":
    main()
