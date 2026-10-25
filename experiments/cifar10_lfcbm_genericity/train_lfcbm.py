#!/usr/bin/env python
"""LF-CBM trainer. --stage a: concept layer; --stage b: one GLM-SAGA chunk
(exit code 0 always; prints DONE when the head is finished)."""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.lfcbm.train import run_stage_a, run_stage_b_chunk

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b"], required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if args.stage == "a":
        run_stage_a(cfg, HERE)
    else:
        done = run_stage_b_chunk(cfg, HERE)
        print("HEAD_DONE" if done else "HEAD_CONTINUE")
