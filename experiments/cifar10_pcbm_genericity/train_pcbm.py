#!/usr/bin/env python
"""PCBM trainer.
  --stage a            build concept bank(s) + standardizer + L2 sanity head
                       (one GPU job builds both banks; needs PCBM_BRODEN_DIR)
  --stage b --bank X   faithful PCBM reference head (sparse elastic-net via
                       glm_saga) for bank X; one <=25-min resumable chunk,
                       prints HEAD_DONE when finished."""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.pcbm.train import (
    run_refresh_proj,
    run_stage_a,
    run_stage_a_mm_large,
    run_stage_b_chunk,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "aL", "b", "refresh"], required=True)
    ap.add_argument("--bank", choices=["cav", "mm", "mmL"], default="cav")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if args.stage == "a":
        run_stage_a(cfg, HERE)
    elif args.stage == "aL":
        run_stage_a_mm_large(cfg, HERE)
    elif args.stage == "refresh":
        cfg["bank"] = args.bank
        run_refresh_proj(cfg, HERE)
    else:
        cfg["bank"] = args.bank
        done = run_stage_b_chunk(cfg, HERE)
        print("HEAD_DONE" if done else "HEAD_CONTINUE")
