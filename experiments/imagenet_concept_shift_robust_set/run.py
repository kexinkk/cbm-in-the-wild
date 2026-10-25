#!/usr/bin/env python
"""§3.4 Concept-shift robust set — compute runner (CPU-only).

Streams all 50 ImageNet-C per-image cells (~25 GB torch.load) + the clean
reference to produce the per-concept shift table, confound diagnostic, robust /
fragile sets (ranked by d_k), and the head-masking ablation. No GPU / CLIP /
SAM2 — cached dumps only.

This is the canonical producer. The notebook
(`notebooks/concept_shift_robust_set.ipynb`) is presentation-only: it loads the
artifacts written here and re-renders. Outputs land in this script's directory.

"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cbm_in_the_wild.paths import CHECKPOINT_DIR
from cbm_in_the_wild.dcbm.persistence import load_cbm_artifacts
from cbm_in_the_wild.dcbm.analysis import load_per_image_cell

# --- config -----------------------------------------------------------------
HERE       = Path(__file__).resolve().parent
REPO       = HERE.parents[1]
DATASET    = "imagenet"
CHECKPOINT = "imagenet_baseline"
PROP_ROOT  = REPO / "experiments/imagenet_c_propagation"
OUTPUT_DIR = HERE
NAMES_FILE = "concept_names_vocab.json"   # open-vocab (CLIP-Dissect) names
FAMILIES = ["fog", "gaussian_noise", "motion_blur", "patch_masking", "salt_and_pepper",
            "jpeg_compression", "row_column_dropout", "random_pixel_dropout",
            "sparse_random_occlusion", "structured_pixel_dropout"]
SEVERITIES = [1, 2, 3, 4, 5]
REPORT_N   = 20
SEED       = 0


def main() -> None:
    torch.set_grad_enabled(False)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    # --- load artifacts + clean reference -----------------------------------
    ckpt = CHECKPOINT_DIR / CHECKPOINT
    art  = load_cbm_artifacts(ckpt)
    C    = art["centroids"].float()                      # (K, D)
    K, D = C.shape
    NORMSQ = (C ** 2).sum(1, keepdim=True).t()           # (1, K)
    sc   = art["scaler"]
    MU   = torch.tensor(sc.mean_, dtype=torch.float32)
    SD   = torch.tensor(sc.scale_, dtype=torch.float32).clamp_min(1e-12)
    W    = art["linear_head"].linear.weight.detach().float()   # (Cls, K)
    B    = art["linear_head"].linear.bias.detach().float()     # (Cls,)

    cdir = ckpt / "clean_propagation"
    CLEAN_ACTS = torch.load(cdir / "concept_act.pt", map_location="cpu").float()   # (N, K)
    CLEAN_EMB  = torch.load(cdir / "clip_emb.pt",   map_location="cpu").float()    # (N, D)
    LABELS     = torch.load(cdir / "labels.pt",     map_location="cpu").long()     # (N,)
    _ids       = torch.load(cdir / "image_ids.pt",  map_location="cpu")
    CLEAN_IDS  = _ids.tolist() if isinstance(_ids, torch.Tensor) else list(_ids)
    POS        = {iid: i for i, iid in enumerate(CLEAN_IDS)}

    try:
        NAMES = json.loads((ckpt / NAMES_FILE).read_text())["names"]
    except FileNotFoundError:
        NAMES = [f"c{k}" for k in range(K)]

    # d_k is computed on the L1 PRE-softmax score raw_k = z·C_kᵀ/‖C_k‖² (decision
    # 2026-06-30): the supervisor's per-concept shift is a magnitude, and magnitude
    # is NOT softmax-invariant (only rank metrics are). Post-softmax over K=2048 is
    # razor-peaked (O(1e-5) shifts, all concepts coupled through the normalizer);
    # pre-softmax scores are O(0.1), signed, per-concept independent. The head-masking
    # ablation (§F) still runs the TRUE forward path (softmax→scaler→head) for valid
    # accuracies — only the concept-shift metric moves pre-softmax.
    def raw_from_emb(z):
        return (z @ C.t()) / NORMSQ            # (N, K) pre-softmax per-concept score

    def logits_from_acts(a):
        return ((a - MU) / SD) @ W.t() + B

    def align_clean(cids):
        return torch.tensor([POS[i] for i in cids], dtype=torch.long)

    CLEAN_RAW   = raw_from_emb(CLEAN_EMB)                 # (N, K) pre-softmax clean scores

    clean_acc = float((logits_from_acts(CLEAN_ACTS).argmax(1) == LABELS).float().mean())
    print(f"K={K} D={D} classes={W.shape[0]} cleanN={CLEAN_ACTS.shape[0]} clean_acc={clean_acc:.4f}")

    # --- A. streaming pass: d_k, direction-only d_k^dir, severity grid -------
    dk_grid    = np.full((K, len(FAMILIES), len(SEVERITIES)), np.nan, dtype=np.float64)
    dkdir_grid = np.full((K, len(FAMILIES), len(SEVERITIES)), np.nan, dtype=np.float64)
    reg_pts    = []   # (Δ||z||, ||Δa||) sampled from sev-3 cells for the confound regression

    for fi, fam in enumerate(FAMILIES):
        for si, sev in enumerate(SEVERITIES):
            pdir = PROP_ROOT / f"{fam}_s{sev}" / "per_image"
            try:
                cell = load_per_image_cell(pdir, fam, sev)
            except FileNotFoundError:
                print(f"  skip missing {fam}_s{sev}"); continue
            cids = cell["image_ids"]
            cids = cids.tolist() if isinstance(cids, torch.Tensor) else list(cids)
            idx  = align_clean(cids)
            cl_r = CLEAN_RAW.index_select(0, idx)         # pre-softmax clean score
            cl_e = CLEAN_EMB.index_select(0, idx)
            ce   = cell["clip_emb"].float()
            cr   = raw_from_emb(ce)                        # pre-softmax corrupt score
            dk_grid[:, fi, si] = (cr - cl_r).abs().mean(0).numpy()
            ce_dir = ce * (cl_e.norm(dim=1, keepdim=True) / ce.norm(dim=1, keepdim=True))
            cr_dir = raw_from_emb(ce_dir)
            dkdir_grid[:, fi, si] = (cr_dir - cl_r).abs().mean(0).numpy()
            if sev == 3:
                g = torch.Generator().manual_seed(SEED)
                s = torch.randperm(cr.shape[0], generator=g)[:2000]
                dz = (ce.norm(dim=1) - cl_e.norm(dim=1))[s].numpy()
                da = (cr - cl_r).norm(dim=1)[s].numpy()
                reg_pts.append(np.stack([dz, da], 1))
            del cell, ce, cl_r, cl_e, cr, cr_dir
        print(f"done {fam}")
    print("grid filled:", np.isfinite(dk_grid).all())

    # --- B. per-concept table + companions ----------------------------------
    # Headline is raw d_k (supervisor's metric): the single robustness ranking
    # used downstream (§3.4.3 robust/fragile sets, §3.4.4/§3.4.5 pruning).
    dk_overall    = np.nanmean(dk_grid, axis=(1, 2))     # (K,)
    dkdir_overall = np.nanmean(dkdir_grid, axis=(1, 2))

    df = pd.DataFrame({
        "concept_id": np.arange(K),
        "name": NAMES,
        "d_k": dk_overall,
        "d_k_dir": dkdir_overall,
        "dir_frac": dkdir_overall / dk_overall,
    })

    lb_path = REPO / "experiments/concept_retention/imagenet/leaderboard.json"
    if lb_path.exists():
        lb = json.loads(lb_path.read_text())
        comp = {}
        for bucket in ("most_robust", "least_robust"):
            for r in lb.get("global_shift_ranking", {}).get(bucket, []):
                comp[r["concept_id"]] = r
        df["retention@5"] = df.concept_id.map(lambda c: comp.get(c, {}).get("retention@5"))
        df["phi_drift"]   = df.concept_id.map(lambda c: comp.get(c, {}).get("phi_drift"))

    # --- C. confound diagnostic ---------------------------------------------
    dir_frac_mean = float(np.nanmean(dkdir_overall / dk_overall))
    pts = np.concatenate(reg_pts, 0)
    r = float(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(dk_overall, bins=60, alpha=.5, label="raw  d_k")
    ax[0].hist(dkdir_overall, bins=60, alpha=.5, label="direction-only  d_k^dir")
    ax[0].set_title(f"shift survives norm control\nmean d_k_dir/d_k = {dir_frac_mean:.2f}")
    ax[0].set_xlabel("d_k"); ax[0].legend()
    ax[1].scatter(pts[:, 0], pts[:, 1], s=3, alpha=.2)
    ax[1].set_title(f"per-image  ||Δraw|| vs Δ||z||   r={r:.2f}  R²={r**2:.2f}")
    ax[1].set_xlabel("Δ||z||"); ax[1].set_ylabel("||Δraw||")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures/C_confound_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"VERDICT: {dir_frac_mean:.0%} of d_k survives holding ||z|| fixed; "
          f"per-image Δ||z|| explains R²={r**2:.2f} of ||Δa||.")

    # --- D. distributions, per-family heatmap, severity dose-response -------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(dk_overall, bins=80); ax[0].set_title("d_k distribution (all concepts)"); ax[0].set_xlabel("d_k")
    order = np.argsort(dk_overall)
    hm = np.nanmean(dk_grid, axis=2)[order]
    im = ax[1].imshow(hm, aspect="auto", cmap="magma")
    ax[1].set_xticks(range(len(FAMILIES))); ax[1].set_xticklabels(FAMILIES, rotation=90, fontsize=7)
    ax[1].set_title("d_k per family (concepts sorted)"); ax[1].set_ylabel("concept rank")
    plt.colorbar(im, ax=ax[1])
    for fi, fam in enumerate(FAMILIES):
        ax[2].plot(SEVERITIES, np.nanmean(dk_grid[:, fi, :], axis=0), marker="o", label=fam, lw=1)
    ax[2].set_title("severity dose-response (mean over concepts)")
    ax[2].set_xlabel("severity"); ax[2].set_ylabel("mean d_k"); ax[2].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures/D_distributions_heatmap_doseresponse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- E. robust / fragile sets by d_k ------------------------------------
    df["w_imp"]  = W.abs().max(0).values.numpy()
    robust  = df.sort_values("d_k").head(REPORT_N)
    fragile = df.sort_values("d_k", ascending=False).head(REPORT_N)

    # --- F. validation ablation: head masked to top-r robust concepts -------
    rank_by_robust = df.sort_values("d_k").concept_id.values
    sizes = [25, 50, 100, 200, 400, 800, 1500, K]

    def masked_acc_on(acts, labels, keep_idx):
        Wm = torch.zeros_like(W); Wm[:, keep_idx] = W[:, keep_idx]
        logit = ((acts - MU) / SD) @ Wm.t() + B
        return (logit.argmax(1) == labels).float().mean().item()

    clean_curve = [masked_acc_on(CLEAN_ACTS, LABELS, torch.tensor(rank_by_robust[:rr])) for rr in sizes]

    corrupt_curve = np.zeros(len(sizes)); ncells = 0
    for fam in FAMILIES:
        pdir = PROP_ROOT / f"{fam}_s3" / "per_image"
        try:
            cell = load_per_image_cell(pdir, fam, 3)
        except FileNotFoundError:
            print(f"  §F skip missing {fam}_s3"); continue
        cids = cell["image_ids"]
        cids = cids.tolist() if isinstance(cids, torch.Tensor) else list(cids)
        lab  = LABELS.index_select(0, align_clean(cids))
        ca   = cell["concept_acts"].float()
        for j, rr in enumerate(sizes):
            corrupt_curve[j] += masked_acc_on(ca, lab, torch.tensor(rank_by_robust[:rr]))
        ncells += 1; del cell, ca
    corrupt_curve /= ncells

    fig = plt.figure(figsize=(6, 4))
    plt.plot(sizes, clean_curve, "o-", label="clean")
    plt.plot(sizes, corrupt_curve, "s-", label="corrupt (sev-3 mean)")
    plt.axhline(clean_curve[-1], ls=":", c="gray"); plt.axhline(corrupt_curve[-1], ls=":", c="gray")
    plt.xscale("log"); plt.xlabel("robust-set size r (most-robust concepts kept)"); plt.ylabel("accuracy")
    plt.title("head masked to top-r robust concepts"); plt.legend(); plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures/F_ablation_robust_set.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    ablation = pd.DataFrame({"r": sizes, "clean": clean_curve, "corrupt": corrupt_curve})

    # --- persist ------------------------------------------------------------
    df.to_csv(OUTPUT_DIR / "concept_shift_table.csv", index=False)
    ablation.to_csv(OUTPUT_DIR / "ablation_robust_set.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "shift_grids.npz",
        dk_grid=dk_grid, dkdir_grid=dkdir_grid,
        families=np.array(FAMILIES), severities=np.array(SEVERITIES),
        reg_pts=pts,
    )
    summary = {
        "dataset": DATASET, "checkpoint": CHECKPOINT,
        "K": int(K), "D": int(D), "clean_N": int(CLEAN_ACTS.shape[0]),
        "n_families": len(FAMILIES), "n_severities": len(SEVERITIES),
        "grid_complete": bool(np.isfinite(dk_grid).all()),
        "clean_acc": clean_acc,
        "confound": {"dir_frac_mean": dir_frac_mean, "delta_z_R2": float(r ** 2)},
        "ablation": {"sizes": sizes, "clean": list(map(float, clean_curve)),
                     "corrupt": list(map(float, corrupt_curve))},
        "robust_top": robust[["concept_id", "name", "d_k"]].head(10).to_dict("records"),
        "fragile_top": fragile[["concept_id", "name", "d_k", "w_imp"]].head(10).to_dict("records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote artifacts to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
