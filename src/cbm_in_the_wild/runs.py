"""Run artifact writer.

Knows nothing about DCBM; purely about capturing what every experiment
dir is expected to contain (see experiments/README.md): a config
snapshot, the git commit, SLURM metadata, and machine-readable metrics.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path


def capture_context(repo_root: Path) -> dict:
    """Return a dict of git + slurm + host info. All fields best-effort."""
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    return {
        "git_commit": git_commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_node": os.environ.get("SLURMD_NODENAME", "local"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "local"),
        "host": socket.gethostname(),
    }


def write_summary(
    path: Path,
    *,
    config: dict,
    metrics: dict,
    timing_seconds: dict,
    context: dict,
    status: str = "ok",
    extra: dict | None = None,
) -> None:
    """Write `summary.json` at `path` and `git_commit.txt` alongside it.

    `extra` is merged into the top level verbatim so runners can add
    run-specific fields (e.g. `target_test_accuracy`, verdict booleans)
    without forcing them into this schema.
    """
    path = Path(path)
    summary = {
        "status": status,
        "config": config,
        "context": context,
        "timing_seconds": timing_seconds,
        "metrics": metrics,
    }
    if extra:
        summary.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    commit_file = path.parent / "git_commit.txt"
    commit_file.write_text(context.get("git_commit", "unknown") + "\n")
