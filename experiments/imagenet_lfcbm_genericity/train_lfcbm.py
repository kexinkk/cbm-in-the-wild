#!/usr/bin/env python
"""LF-CBM trainer. --stage a: concept layer; --stage b: one GLM-SAGA chunk
(exit code 0 always; prints HEAD_DONE when the head is finished).
--seed N (added 2026-10-11, 3-seed rule): same recipe into model_seed<N>/; without --seed the
original single model in model/ (seed from config.yaml). --stage eval: clean val/test top-1 of
model/ and every finished model_seed* -> seeds.json (deployed = best clean test top-1)."""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.lfcbm.train import run_stage_a, run_stage_b_chunk


def evaluate(cfg):
    from cbm_in_the_wild.lfcbm.core import load_artifacts, load_emb_split
    data = {s: load_emb_split(cfg["dataset"], s) for s in ("val", "test")}
    res = {}
    for md in [HERE / "model"] + sorted(HERE.glob("model_seed*")):
        if not (md / "head_meta.json").exists():
            continue
        art = load_artifacts(md)
        W, b = torch.load(md / "W_g.pt").float(), torch.load(md / "b_g.pt").float()
        r = {"K": int(art["W_c"].shape[0])}
        for split, (_, X, y, _) in data.items():
            acc = 0
            for i in range(0, len(X), 50_000):
                c = (X[i:i + 50_000].float() @ art["W_c"].float().t() - art["proj_mean"]) / art["proj_std"]
                acc += int(((c @ W.t() + b).argmax(1) == y[i:i + 50_000]).sum())
            r[f"{split}_top1"] = acc / len(y)
        res[md.name] = r
    seeds = {k: v for k, v in res.items() if k.startswith("model_seed")}
    out = {"models": res, "rule": "best clean test top-1 (author rule 2026-10-04)",
           "deployed": max(seeds, key=lambda k: seeds[k]["test_top1"]) if seeds else None}
    (HERE / "seeds.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "eval"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if args.seed is not None:
        cfg["seed"], cfg["model_subdir"] = args.seed, f"model_seed{args.seed}"
    if args.stage == "a":
        run_stage_a(cfg, HERE)
    elif args.stage == "b":
        done = run_stage_b_chunk(cfg, HERE)
        print("HEAD_DONE" if done else "HEAD_CONTINUE")
    else:
        evaluate(cfg)
