"""LF-CBM: capacity-matched prune-vs-random ablation.

Protocol mirrors experiments/imagenet_concept_shift_robust_set/
ablation_capacity_matched.py (val-refit): at each size K keep either the
K lowest-d_k concepts (dk_hold3 = primary, sev-3 excluded → no leakage
into the sev-3 eval; dk = all-sev secondary) or a random K-subset
(rand0-2 = selection-null band), refit a fresh scaler + L2 logistic head
on a class-stratified held-out-val subsample, then eval clean (full test)
and corrupt (sev-3 mean over the 10 families). No native-KMeans arm:
LF-CBM concepts are text-derived, not cluster centroids, so "build small"
has no direct analog (noted in notes.md).

d_k is the raw projection shift (concept_shift.py, DCBM-comparable). A
dk_std_hold3 arm ranks on the recorded standardized twin so the figure
shows whether the ranking space changes the verdict (the two rankings
diverge in the low-d_k tail; see concept_shift.py docstring).

Differences from the DCBM script, both deliberate: the head refit is a
full-batch LBFGS multinomial logistic on GPU matching sklearn
LogisticRegression(C=1.0)'s objective (sum-CE + 0.5·||W||²), because these
jobs are confined to 30-min GPU partitions; and pruning selects columns
of the standardized activations directly (LF-CBM has no concept softmax
whose support could change). Ranking lives in raw-projection space; the
head still reads standardized activations, exactly as in DCBM (ranks on
raw d_k, forward path runs softmax→scaler→head).

Resumable via results.jsonl; each invocation stops cleanly before the
30-min wall (TIME_BUDGET_S) and afterany waves finish the grid.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cells import load_cell
from .core import load_artifacts, load_emb_split, std_acts_from_emb

T0 = time.time()
TIME_BUDGET_S = int(os.environ.get("TIME_BUDGET_S", 1290))


def _time_left() -> float:
    return TIME_BUDGET_S - (time.time() - T0)


def _stratified_idx(y: torch.Tensor, n_total: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    classes = y.unique()
    per = max(1, n_total // len(classes))
    picks = []
    for c in classes:
        pool = (y == c).nonzero(as_tuple=True)[0]
        picks.append(pool[torch.randperm(len(pool), generator=g)[:per]])
    return torch.cat(picks)


def _fit_head(Xf, yf, n_classes: int, device: str, max_iter: int = 800):
    """sklearn LogisticRegression(C=1.0)-equivalent multinomial logistic."""
    Xf, yf = Xf.to(device), yf.to(device)
    W = torch.zeros(n_classes, Xf.shape[1], device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xf @ W.t() + b, yf, reduction="sum") \
            + 0.5 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _acc(X, y, W, b, mean, std, device, bs=100_000) -> float:
    hits = 0
    for i in range(0, len(X), bs):
        xb = ((X[i : i + bs].to(device).float() - mean) / std)
        hits += int((xb @ W.t() + b).argmax(1).eq(y[i : i + bs].to(device)).sum())
    return hits / len(X)


def run(cfg: dict, exp_dir: Path) -> None:
    device = "cuda"
    art = load_artifacts(exp_dir / "model", device=device)
    K_full = art["W_c"].shape[0]
    n_classes = art["W_g"].shape[0]

    # rankings
    import pandas as pd

    tbl = pd.read_csv(exp_dir / "concept_shift_table.csv")
    rankings = {
        "dk_hold3": np.argsort(tbl["d_k_hold3"].values),        # primary (raw)
        "dk": np.argsort(tbl["d_k"].values),                    # raw, all-sev
        "dk_std_hold3": np.argsort(tbl["d_k_std_hold3"].values),  # std reference
    }
    if os.environ.get("DK_DISJOINT_TABLE"):
        # d_k (all severities) re-estimated on a train pool disjoint from the eval images
        # (experiments/imagenet_dk_disjoint_pool/compute_dk.py)
        dj = pd.read_csv(os.environ["DK_DISJOINT_TABLE"])
        assert len(dj) == K_full, (len(dj), K_full)
        rankings["dk_disjoint"] = np.argsort(dj["d_k"].values)
    if os.environ.get("DK_LOCO_GRID"):
        # leave-one-corruption-out: d_k over the other 9 corruptions (all severities)
        g = np.load(os.environ["DK_LOCO_GRID"])
        fams = [str(f) for f in g["families"]]
        assert g["dk_grid"].shape[0] == K_full, (g["dk_grid"].shape, K_full)
        for i, held in enumerate(fams):
            keep = [j for j in range(len(fams)) if j != i]
            rankings[f"dk_loco_{held}"] = np.argsort(np.nanmean(g["dk_grid"][:, keep, :], axis=(1, 2)))
    for r in range(3):
        rankings[f"rand{r}"] = np.random.RandomState(1000 + r).permutation(K_full)

    sizes = [s for s in cfg["ablation"]["sizes"] if s < K_full] + [K_full]
    # optional restriction (LOCO runs): ABL_RANKS / ABL_SIZES, comma-separated
    if os.environ.get("ABL_RANKS"):
        rankings = {r: rankings[r] for r in os.environ["ABL_RANKS"].split(",") if r}
    if os.environ.get("ABL_SIZES"):
        sizes = [int(s) for s in os.environ["ABL_SIZES"].split(",") if s]
    seeds = list(cfg["ablation"]["seeds"])
    fit_n = int(cfg["ablation"]["fit_n"])
    _pool_warned = False

    grid = [(rk, K, sd) for K in sizes for rk in rankings for sd in seeds]
    # ABL_OUT_DIR: write results/CSV/summary elsewhere (LOCO), model + cells still from exp_dir
    out_dir = Path(os.environ.get("ABL_OUT_DIR", str(exp_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "ablation_results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["rank"], r["K"], r["seed"]))
    todo = [g for g in grid if g not in done]
    print(f"[abl] grid {len(grid)}, done {len(done)}, todo {len(todo)}")
    if not todo:
        _summarize(cfg, out_dir, res_path)
        return

    # fit side: held-out val; eval side: clean test + sev-3 corrupt cells
    # Refit pool split. Default "val" preserves the ImageNet/CIFAR protocol
    # (25k stratified from the 128k held-out val pool). CUB sets "train":
    # its val split is only 600 images for 200 classes, far too small to
    # fit a K-dim 200-way head (01_CUB.md 5b). Eval stays on test either way.
    fit_split = str(cfg["ablation"].get("fit_split", "val"))
    _, Xva, yva, _ = load_emb_split(cfg["dataset"], fit_split)
    if len(yva) < fit_n:
        print(f"[WARN] ablation refit pool is short: {fit_split} split has "
              f"{len(yva)} images but fit_n={fit_n} requested -> fitting on "
              f"{len(yva)}. Heads will be undertrained and prune-vs-random "
              f"margins inflated by the fit regime.", flush=True)
    test_ids, Xte, yte, _ = load_emb_split(cfg["dataset"], "test")
    row_of = {i: k for k, i in enumerate(test_ids)}
    with torch.no_grad():
        val_acts = std_acts_from_emb(Xva.to(device), art).cpu()
        clean_acts = std_acts_from_emb(Xte.to(device), art).half().cpu()

    from cbm_in_the_wild.corruption_eval import resolve_corruptions

    corrupt = {}
    for fam in resolve_corruptions(cfg["corruption_set"]):
        cell = load_cell(cfg, exp_dir, fam, 3)
        with torch.no_grad():
            a = std_acts_from_emb(cell["clip_emb"].to(device), art).half().cpu()
        idx = torch.tensor([row_of[i] for i in cell["image_ids"]], dtype=torch.long)
        corrupt[fam] = (a, yte[idx])
    print(f"[abl] cached {len(corrupt)} sev-3 cells")

    with open(res_path, "a") as f:
        for rk, K, sd in todo:
            if _time_left() < 120:
                print(f"[abl] time budget reached, exiting for next wave")
                return
            t0 = time.time()
            keep = torch.tensor(np.sort(rankings[rk][:K]), dtype=torch.long)
            fit_idx = _stratified_idx(yva, fit_n, seed=sd)
            Xf = val_acts[fit_idx][:, keep]
            mean = Xf.mean(0).to(device)
            std = Xf.std(0).clamp_min(1e-6).to(device)
            Xf = (Xf.to(device) - mean) / std
            W, b = _fit_head(Xf, yva[fit_idx], n_classes, device)

            clean_acc = _acc(clean_acts[:, keep], yte, W, b, mean, std, device)
            corr = [
                _acc(a[:, keep], yc, W, b, mean, std, device)
                for a, yc in corrupt.values()
            ]
            row = {
                "rank": rk, "K": int(K), "seed": int(sd),
                "clean_acc": clean_acc,
                "corrupt_acc_sev3": float(np.mean(corr)),
                "corrupt_acc_per_family": {
                    fam: c for fam, c in zip(corrupt.keys(), corr)
                },
                "fit_n": len(fit_idx), "fit_split": fit_split, "wall_s": round(time.time() - t0, 1),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[abl] {rk} K={K} seed={sd} clean {clean_acc:.4f} "
                  f"corrupt {row['corrupt_acc_sev3']:.4f} ({row['wall_s']}s)")
    _summarize(cfg, out_dir, res_path)


def _summarize(cfg: dict, exp_dir: Path, res_path: Path) -> None:
    import pandas as pd

    rows = [json.loads(l) for l in res_path.read_text().splitlines()]
    df = pd.DataFrame(rows).drop(columns=["corrupt_acc_per_family"])
    df.to_csv(exp_dir / "ablation_capacity_matched.csv", index=False)
    agg = df.groupby(["rank", "K"]).agg(
        corrupt_mean=("corrupt_acc_sev3", "mean"),
        corrupt_std=("corrupt_acc_sev3", "std"),
        clean_mean=("clean_acc", "mean"),
    ).reset_index()

    out = {"n_fits": len(df), "cells": agg.to_dict("records"), "margins": {}}
    for K in sorted(df["K"].unique()):
        sub = agg[agg["K"] == K].set_index("rank")
        rand = [sub.loc[f"rand{r}", "corrupt_mean"]
                for r in range(3) if f"rand{r}" in sub.index]
        if rand and "dk_hold3" in sub.index:
            out["margins"][int(K)] = {
                "dk_hold3_vs_random_pp": float(
                    (sub.loc["dk_hold3", "corrupt_mean"] - np.mean(rand)) * 100),
                "random_band": [float(min(rand)), float(max(rand))],
            }
    (exp_dir / "ablation_capacity_matched_summary.json").write_text(
        json.dumps(out, indent=2))
    print("[abl] summary written:", json.dumps(out["margins"], indent=1))
