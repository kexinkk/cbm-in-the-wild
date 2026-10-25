#!/usr/bin/env python
"""Capacity-matched prune-vs-random (resumable waves).
--bank selects the concept source (cav|mm)."""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.pcbm.ablation import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["cav", "mm", "mmL"], default="cav")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    cfg["bank"] = args.bank
    run(cfg, HERE)
