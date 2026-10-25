#!/usr/bin/env python
"""One-off: embed corrupted CIFAR-10 test cells (50-task SLURM array)."""
import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.lfcbm.cells import generate_cifar_cell

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if args.task_id >= 0:
        generate_cifar_cell(cfg, HERE, args.task_id)
    else:
        for t in range(50):
            generate_cifar_cell(cfg, HERE, t)
