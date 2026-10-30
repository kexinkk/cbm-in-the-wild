#!/usr/bin/env python
"""Retrain LF-CBM / PCBM-mmL heads on d_k-pruned concept sets with each family's own recipe.

Companion of experiments/imagenet_pruned_bank_deployed_retrain (DCBM). The concept
layer (LF-CBM W_c, PCBM-mmL text bank) and its train-set standardization stay the
deployed ones; only the kept columns change. The head is then trained exactly like
the deployed head (lfcbm/train.py, pcbm/train.py stage B): GLM-SAGA elastic net
(alpha 0.99, step 0.1) on the 1.15M train split, lam / epochs from the family
config, zero init, final path point.

Arms: full | dk<K> (paper ranking, all-severity d_k) | rand<K> (draw = the paper's
RandomState(1000+seed)) | dkd<K> (d_k on the disjoint train pool) | loco_<c>_<K>
(disjoint-pool d_k over the 9 corruptions other than c). Seeds 0-2 set
the SAGA shuffle (and the random draw).

    python run.py --family lfcbm --arm dk800 --seed 0     # one resumable chunk
    python run.py --summarize

Each call trains until TIME_BUDGET_S seconds (default 1290), persisting glm_saga state
under $DCBM_DATA_DIR/pruned_family_retrain; rerun the same command to continue. When the
head is done it scores clean (full 50k test) + all 50 corrupted cells and writes
runs/<family>/<arm>_seed<seed>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cbm_in_the_wild.paths import DATA_DIR
from cbm_in_the_wild.seeding import set_all_seeds

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DISJ_DIR = REPO / "experiments/imagenet_dk_disjoint_pool"
DEVICE = "cuda"
T0 = time.time()

FAMILY = {
    "lfcbm": dict(exp="imagenet_lfcbm_genericity", model="model",
                  table="concept_shift_table.csv", disj="lfcbm_concept_shift_table.csv"),
    "pcbm_mmL": dict(exp="imagenet_pcbm_genericity", model="model_mmL",
                     table="mmL_concept_shift_table.csv", disj="pcbm_mmL_concept_shift_table.csv"),
}


def _core(family):
    if family == "lfcbm":
        from cbm_in_the_wild.lfcbm import core
    else:
        from cbm_in_the_wild.pcbm import core
    return core


def split_arm(arm: str):
    if arm.startswith("loco_"):                          # loco_<held corruption>_<K>
        held, K = arm.removeprefix("loco_").rsplit("_", 1)
        return f"loco_{held}", int(K)
    for p in ("full", "dkd", "dk", "rand"):
        if arm.startswith(p):
            return p, (int(arm[len(p):]) if arm[len(p):] else None)
    raise ValueError(arm)


def keep_ids(family: str, arm: str, seed: int, K_full: int) -> np.ndarray:
    f = FAMILY[family]
    pre, K = split_arm(arm)
    if pre == "full":
        return np.arange(K_full)
    if pre == "dk":
        r = np.argsort(pd.read_csv(REPO / "experiments" / f["exp"] / f["table"])["d_k"].values)
    elif pre == "dkd":
        r = np.argsort(pd.read_csv(DISJ_DIR / f["disj"])["d_k"].values)
    elif pre.startswith("loco_"):                        # disjoint-pool d_k without corruption c
        z = np.load(DISJ_DIR / f["disj"].replace("concept_shift_table.csv", "shift_grids.npz"))
        held = pre.removeprefix("loco_")
        keep = [i for i, c in enumerate(z["families"]) if str(c) != held]
        assert len(keep) == len(z["families"]) - 1, f"unknown corruption {held!r}"
        r = np.argsort(np.nanmean(z["dk_grid"][:, keep, :], axis=(1, 2)))
    else:
        r = np.random.RandomState(1000 + seed).permutation(K_full)
    return np.sort(r[:K])


def sliced_art(art: dict, keep: torch.Tensor) -> dict:
    """Deployed concept layer restricted to the kept columns (stats unchanged)."""
    out = dict(art)
    for k in ("W_c", "cavs"):
        if k in art:
            out[k] = art[k][keep]
    if "norms" in art:
        out["norms"] = art["norms"][keep]
    out["proj_mean"] = art["proj_mean"][..., keep]
    out["proj_std"] = art["proj_std"][..., keep]
    return out


def project(core, X, art):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 100_000):
            outs.append(core.std_acts_from_emb(X[i:i + 100_000].to(DEVICE), art).cpu())
    return torch.cat(outs)


def train(family, arm, seed, cfg) -> dict | None:
    from cbm_in_the_wild.lfcbm.core import add_lfcbm_to_path
    add_lfcbm_to_path()
    from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

    core = _core(family)
    f = FAMILY[family]
    saga = cfg["saga"][family]
    art_full = core.load_artifacts(REPO / "experiments" / f["exp"] / f["model"], device=DEVICE)
    K_full = art_full["proj_mean"].shape[-1]
    keep_np = keep_ids(family, arm, seed, K_full)
    keep = torch.as_tensor(keep_np, dtype=torch.long, device=DEVICE)
    art = sliced_art(art_full, keep)

    state_dir = DATA_DIR / "pruned_family_retrain" / family
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{arm}_seed{seed}.saga_state.pt"
    prog_path = state_dir / f"{arm}_seed{seed}.progress.json"

    set_all_seeds(seed)
    _, Xtr, ytr, classes = core.load_emb_split("imagenet", "train")
    _, Xva, yva, _ = core.load_emb_split("imagenet", "val")
    train_c, val_c = project(core, Xtr, art), project(core, Xva, art)
    del Xtr, Xva
    print(f"[{family}/{arm}/s{seed}] K={len(keep_np)} train {tuple(train_c.shape)} "
          f"({time.time() - T0:.0f}s)", flush=True)

    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(DEVICE)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    epochs_done, state = 0, None
    if prog_path.exists() and state_path.exists():
        epochs_done = json.loads(prog_path.read_text())["epochs_done"]
        blob = torch.load(state_path, map_location="cpu")
        state = {"a_table": blob["a_table"], "w_grad_avg": blob["w_grad_avg"].to(DEVICE),
                 "b_grad_avg": blob["b_grad_avg"].to(DEVICE)}
        linear.load_state_dict(blob["linear"])
        print(f"resuming at epoch {epochs_done}", flush=True)

    n_total = int(saga["n_iters"])
    loader = DataLoader(IndexedTensorDataset(train_c, ytr), batch_size=int(saga["batch_size"]),
                        shuffle=True)
    val_loader = DataLoader(TensorDataset(val_c, yva), batch_size=int(saga["batch_size"]),
                            shuffle=False)
    budget = int(os.environ.get("TIME_BUDGET_S", 1290))
    per_epoch, params = None, None
    while epochs_done < n_total:
        elapsed = time.time() - T0
        n = 1 if per_epoch is None else max(0, min(
            n_total - epochs_done, int((budget - elapsed - 150) / (per_epoch * 1.25))))
        if n == 0:
            print(f"wall budget reached at {epochs_done}/{n_total}")
            return None
        t0 = time.time()
        out = glm_saga(linear, loader, 0.1, n, 0.99, epsilon=1, k=1,
                       val_loader=val_loader, do_zero=False, verbose=1,
                       metadata={"max_reg": {"nongrouped": float(saga["lam"])}},
                       n_ex=len(train_c), n_classes=len(classes), state=state)
        per_epoch = (time.time() - t0) / n
        state, params = out["state"], out["path"][0]
        epochs_done += n
        tmp = state_path.with_suffix(".tmp")
        torch.save({"a_table": state["a_table"].cpu(), "w_grad_avg": state["w_grad_avg"].cpu(),
                    "b_grad_avg": state["b_grad_avg"].cpu(),
                    "linear": {k: v.cpu() for k, v in linear.state_dict().items()}}, tmp)
        tmp.rename(state_path)
        prog_path.write_text(json.dumps({"epochs_done": epochs_done}))
        print(f"+{n} ep (~{per_epoch:.0f}s/ep) {epochs_done}/{n_total} "
              f"val_acc {params['metrics']['acc_val']:.4f}", flush=True)

    if params is None:                                   # finished in an earlier segment
        params = {"weight": linear.weight.detach(), "bias": linear.bias.detach(),
                  "metrics": {"acc_val": float("nan")}}
    W, b = params["weight"].to(DEVICE).float(), params["bias"].to(DEVICE).float()
    return {"core": core, "art": art, "keep": keep_np, "W": W, "b": b,
            "val_acc": float(params["metrics"]["acc_val"]), "epochs": epochs_done,
            "nnz": int((W.abs() > 1e-5).sum()), "state_path": state_path, "prog_path": prog_path}


@torch.no_grad()
def evaluate(family, h, cfg) -> dict:
    from cbm_in_the_wild.lfcbm.cells import load_cell
    core, art = h["core"], h["art"]
    ids, Xte, yte, _ = core.load_emb_split("imagenet", "test")
    row = {i: k for k, i in enumerate(ids)}

    def acc(z, y):
        logits = core.std_acts_from_emb(z.to(DEVICE).float(), art) @ h["W"].t() + h["b"]
        return float((logits.argmax(1).cpu() == y).float().mean())

    cells = {"clean": acc(Xte, yte)}
    exp = REPO / "experiments" / FAMILY[family]["exp"]
    for c in cfg["corruptions"]:
        for s in cfg["severities"]:
            cell = load_cell({"dataset": "imagenet"}, exp, c, s)
            cells[f"{c}_s{s}"] = acc(cell["clip_emb"],
                                     yte[torch.tensor([row[i] for i in cell["image_ids"]])])
    sev = {s: float(np.mean([cells[f"{c}_s{s}"] for c in cfg["corruptions"]]))
           for s in cfg["severities"]}
    return {"clean": cells["clean"], "corrupt_sev3": sev[cfg["headline_severity"]],
            "corrupt_by_sev": sev, "cells": cells, "n_test": len(yte)}


def run_one(family, arm, seed, cfg):
    out = HERE / "runs" / family / f"{arm}_seed{seed}.json"
    if out.exists():
        print(f"skip: {out} exists")
        return
    h = train(family, arm, seed, cfg)
    if h is None:
        return
    ev = evaluate(family, h, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"family": family, "arm": arm, "seed": seed, "K": int(len(h["keep"])),
           "keep_ids": h["keep"].tolist(), "val_acc": h["val_acc"], "epochs": h["epochs"],
           "nnz": h["nnz"], "recipe": cfg["saga"][family], **ev,
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    out.write_text(json.dumps(rec))
    torch.save({"W": h["W"].cpu(), "b": h["b"].cpu(), "keep": h["keep"]},
               h["state_path"].with_name(f"{arm}_seed{seed}.head.pt"))
    h["state_path"].unlink(missing_ok=True)
    print(f"done {family}/{arm}/s{seed}: clean {ev['clean']:.4f} "
          f"corrupt_sev3 {ev['corrupt_sev3']:.4f}")


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted((HERE / "runs").glob("*/*.json"))]
    if not rows:
        print("no finished runs")
        return
    df = pd.DataFrame([{k: r[k] for k in ("family", "arm", "seed", "K", "clean", "corrupt_sev3",
                                          "val_acc", "nnz")} for r in rows])
    df.to_csv(HERE / "results.csv", index=False)
    agg = (df.groupby(["family", "arm", "K"])
             .agg(n=("seed", "count"), clean_m=("clean", "mean"), clean_s=("clean", "std"),
                  corrupt_m=("corrupt_sev3", "mean"), corrupt_s=("corrupt_sev3", "std"))
             .reset_index().sort_values(["family", "K", "arm"]))
    (HERE / "summary.json").write_text(json.dumps(
        {"protocol": "family-recipe GLM-SAGA retrain on pruned concepts; eval full 50k",
         "table": agg.to_dict("records")}, indent=1))
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=list(FAMILY))
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if a.family:
        run_one(a.family, a.arm, a.seed, cfg)
    if a.summarize:
        summarize()
