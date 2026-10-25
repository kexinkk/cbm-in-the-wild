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
    ap.add_argument("--seed", type=int, default=None, help="override config seed (3-seed replicates, 2026-10-11)")
    ap.add_argument("--suffix", default=None, help="model dir suffix, e.g. _seed1 (never the deployed dir)")
    ap.add_argument("--banks", default=None, help="stage a: comma list overriding config banks")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.suffix:
        cfg["model_suffix"] = args.suffix
    if args.banks:
        cfg["banks"] = args.banks.split(",")
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
