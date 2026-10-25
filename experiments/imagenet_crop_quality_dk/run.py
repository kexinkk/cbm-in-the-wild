#!/usr/bin/env python
"""§3.4.3b (T1a) — does crop image quality drive d_k? Correlational pass.

Meeting 2026-07-24: the §3.4.3 montages show robust concepts (low d_k) with higher-
resolution, larger, squarer representative crops, fragile concepts the reverse. Question:
is d_k measuring semantic robustness or crop-geometry statistics?

Unit of analysis = **cluster membership**, not the single montage crop. Each of the 2048
concepts is a centroid; assign every SAM2 crop to its nearest centroid (cosine argmax,
matching the normalized-KMeans geometry), then take the per-concept MEDIAN of three
geometry metrics:
  min_side = min(w,h)   — CLIP resizes the short side to 224, so this is the binding
                          resolution (upscale factor 224/min_side)
  area     = w·h        — pixel count
  AR       = max/min    — elongation, orientation folded out (§3.4.3 "closer to 1")
(w/h kept as a secondary signed metric.) Metrics come from JPEG headers only — no decode.

Outputs (decided 2026-07-24): per metric, one scatter of per-concept median vs d_k over all
2048 concepts with Spearman rho, robust/fragile-20 highlighted; partial rho controlling the
centroid norm ‖C_k‖ (geometry may correlate with d_k through norm); pairwise metric rho (the
three are not independent); and per-corruption-family rho (noise/blur-specific vs generic).

Two phases (auto): if crop_dims.npz is absent, walk the crops tree once (metadata-heavy,
cached forever — a one-off pass), then analyze. Also writes membership.npz +
the fragile-set median min_side that the T1b causal runner (run_causal.py) consumes.

Producer for concept_quality_table.csv + summary.json + figures/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr

from cbm_in_the_wild.paths import CHECKPOINT_DIR, REPO_ROOT
from cbm_in_the_wild.persistence import load_cbm_artifacts

HERE       = Path(__file__).resolve().parent
REPO       = HERE.parents[1]
CHECKPOINT = "imagenet_baseline"
DCBM_REPO  = Path(os.environ.get("DCBM_REPO", REPO_ROOT.parent / "DCBM"))
CROP_BLOB  = DCBM_REPO / "data/Segments/segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch"
CROPS_ROOT = DCBM_REPO / "data/Segments/imagenet_SAM2/crops"
SHIFT_DIR  = REPO / "experiments/imagenet_concept_shift_robust_set"
REPORT_N   = 20
CHUNK      = 50_000


def load_blob():
    """(cropnames, embeddings (N,D) float32, wnids) in key order."""
    blob = torch.load(CROP_BLOB, map_location="cpu", weights_only=False)
    names = list(blob.keys())
    emb = torch.stack([blob[k][0].float() for k in names]).numpy()
    wnids = [blob[k][1] for k in names]
    del blob
    return names, emb, wnids


def ensure_dims(names, wnids) -> tuple[np.ndarray, np.ndarray]:
    """Per-crop (w, h) from JPEG headers (no decode). Cached to crop_dims.npz.

    The one metadata-heavy pass: ~N Image.open()s over the loose crops tree. Cached so
    it never re-runs.
    """
    f = HERE / "crop_dims.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if len(z["names"]) == len(names):
            print(f"loaded cached crop_dims.npz ({len(names)} crops)", flush=True)
            return z["w"].astype(np.float64), z["h"].astype(np.float64)
    print(f"reading {len(names)} JPEG headers from {CROPS_ROOT} …", flush=True)
    w = np.zeros(len(names)); h = np.zeros(len(names))
    miss = 0
    for i, (nm, wn) in enumerate(zip(names, wnids)):
        try:
            with Image.open(CROPS_ROOT / wn / nm) as im:
                w[i], h[i] = im.size            # PIL: (width, height), lazy — no decode
        except (FileNotFoundError, OSError):
            w[i] = h[i] = np.nan; miss += 1
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{len(names)} headers", flush=True)
    if miss:
        print(f"  WARNING: {miss} crops unreadable (dims=nan)", flush=True)
    np.savez(f, names=np.array(names, dtype=object), w=w, h=h)
    return w, h


def partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x,y controlling z (Pearson on rank-transforms)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = (pd.Series(v[ok]).rank().values for v in (x, y, z))
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-12))
    rxy, rxz, ryz = pear(rx, ry), pear(rx, rz), pear(ry, rz)
    return (rxy - rxz * ryz) / (np.sqrt((1 - rxz**2) * (1 - ryz**2)) + 1e-12)


def main() -> None:
    torch.set_grad_enabled(False)
    (HERE / "figures").mkdir(parents=True, exist_ok=True)

    names, emb, wnids = load_blob()
    N = len(names)
    w, h = ensure_dims(names, wnids)

    # per-crop geometry metrics
    min_side = np.minimum(w, h)
    area     = w * h
    ar       = np.maximum(w, h) / np.minimum(w, h)
    wh       = w / h
    METRICS  = {"min_side": min_side, "area": area, "AR": ar, "wh": wh}

    # --- membership: cosine argmax to the 2048 centroids ---------------------
    art = load_cbm_artifacts(CHECKPOINT_DIR / CHECKPOINT)
    C   = art["centroids"].float()
    K, D = C.shape
    Cn  = torch.nn.functional.normalize(C, dim=1)
    E   = torch.from_numpy(emb)
    assign = np.zeros(N, dtype=np.int64)
    best_sim = np.full(N, -np.inf)
    rep_sim = np.full(K, -np.inf); rep_idx = np.zeros(K, dtype=np.int64)   # nearest crop / concept
    for s in range(0, N, CHUNK):
        e = torch.nn.functional.normalize(E[s:s + CHUNK], dim=1)
        sim = e @ Cn.t()                                   # (chunk, K)
        mx, arg = sim.max(dim=1)
        assign[s:s + CHUNK] = arg.numpy()
        best_sim[s:s + CHUNK] = mx.numpy()
        cmx, carg = sim.max(dim=0)                          # best crop per concept in-chunk
        upd = cmx.numpy() > rep_sim
        rep_sim[upd] = cmx.numpy()[upd]; rep_idx[upd] = (carg.numpy() + s)[upd]
    counts = np.bincount(assign, minlength=K)

    # per-concept median of each metric over its members (+ representative-crop metric)
    dfm = pd.DataFrame({"assign": assign, **METRICS})
    med = dfm.groupby("assign").median()
    per_concept = {m: np.full(K, np.nan) for m in METRICS}
    for m in METRICS:
        per_concept[m][med.index.values] = med[m].values
    rep = {f"rep_{m}": METRICS[m][rep_idx] for m in METRICS}

    # --- align with d_k, centroid norm, per-family d_k -----------------------
    dk_df = pd.read_csv(SHIFT_DIR / "concept_shift_table.csv").set_index("concept_id")
    d_k   = dk_df["d_k"].reindex(range(K)).values
    cnorm = (C ** 2).sum(1).sqrt().numpy()
    grids = np.load(SHIFT_DIR / "shift_grids.npz", allow_pickle=True)
    FAMS  = [str(f) for f in grids["families"]]
    dk_fam = np.nanmean(grids["dk_grid"], axis=2)          # (K, n_fam) mean over severities

    # --- correlations --------------------------------------------------------
    corr = {}
    for m in METRICS:
        v = per_concept[m]
        rho, p = spearmanr(v, d_k, nan_policy="omit")
        pr = partial_spearman(v, d_k, cnorm)
        corr[m] = {"spearman_rho": round(float(rho), 4), "p": float(p),
                   "partial_rho_ctrl_norm": round(float(pr), 4),
                   "per_family_rho": {FAMS[fi]: round(float(spearmanr(v, dk_fam[:, fi],
                                      nan_policy="omit")[0]), 4) for fi in range(len(FAMS))}}
    # pairwise metric correlations (the three are not independent evidence)
    keys = list(METRICS)
    pair = {f"{a}~{b}": round(float(spearmanr(per_concept[a], per_concept[b],
            nan_policy="omit")[0]), 4) for i, a in enumerate(keys) for b in keys[i + 1:]}

    # --- scatter figures (3 primary metrics; robust/fragile-20 highlighted) --
    robust = np.argsort(d_k)[:REPORT_N]
    fragile = np.argsort(d_k)[-REPORT_N:]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for j, m in enumerate(["min_side", "area", "AR"]):
        v = per_concept[m]
        ax[j].scatter(v, d_k, s=6, c="0.7", alpha=0.5)
        ax[j].scatter(v[robust], d_k[robust], s=22, c="C0", label="robust-20")
        ax[j].scatter(v[fragile], d_k[fragile], s=22, c="C3", label="fragile-20")
        ax[j].set_xlabel(f"per-concept median {m}"); ax[j].set_ylabel("d_k")
        ax[j].set_title(f"{m}: rho={corr[m]['spearman_rho']:.3f} "
                        f"(partial|‖C‖ {corr[m]['partial_rho_ctrl_norm']:.3f})")
        if m in ("area",):
            ax[j].set_xscale("log")
        ax[j].legend(fontsize=7)
    fig.suptitle("§3.4.3b — crop geometry vs corruption shift d_k (2048 concepts)", y=1.02)
    plt.tight_layout()
    fig.savefig(HERE / "figures/crop_quality_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- per-concept table + membership (for T1b) + summary ------------------
    out = pd.DataFrame({"concept_id": np.arange(K), "d_k": d_k, "c_norm": cnorm,
                        "n_members": counts,
                        **{f"med_{m}": per_concept[m] for m in METRICS}, **rep})
    out.to_csv(HERE / "concept_quality_table.csv", index=False)
    # membership for the causal test: assignment + the fragile-set median min_side target
    frag_min = float(np.nanmedian(per_concept["min_side"][fragile]))
    robust_min = float(np.nanmedian(per_concept["min_side"][robust]))
    np.savez(HERE / "membership.npz",
             names=np.array(names, dtype=object), wnids=np.array(wnids, dtype=object),
             assign=assign, min_side=min_side,
             fragile_median_min_side=frag_min, robust_median_min_side=robust_min)

    summary = {
        "method": "crop_quality_correlational", "checkpoint": CHECKPOINT,
        "K": int(K), "n_crops": int(N), "report_n": REPORT_N,
        "correlations": corr, "pairwise_metric_rho": pair,
        "fragile_median_min_side": round(frag_min, 1),
        "robust_median_min_side": round(robust_min, 1),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Spearman rho(metric, d_k):",
          {m: corr[m]["spearman_rho"] for m in METRICS}, flush=True)
    print("partial rho | ‖C‖:",
          {m: corr[m]["partial_rho_ctrl_norm"] for m in METRICS}, flush=True)
    print(f"fragile median min_side={frag_min:.0f}px  robust={robust_min:.0f}px", flush=True)
    print("wrote concept_quality_table.csv + membership.npz + summary + figure", flush=True)


if __name__ == "__main__":
    main()
