#!/usr/bin/env python
"""§3.4.3b (T1b) — causal test: does degrading crop resolution CAUSE higher d_k?

Intervention on the R=50 lowest-d_k (robust) concepts: downsample every member crop to the
fragile-set median min_side (bicubic, aspect preserved), re-embed with CLIP, rebuild the
centroid, and recompute d_k. Arms:
  intervention : robust-50, members downsampled          — does low res cause high d_k?
  control      : robust-50, members re-embedded unchanged — pipeline noise floor
  general      : mid-d_k-50, members downsampled          — is the effect robust-specific?

d_k is per-concept independent (raw_k(z)=z·C_kᵀ/‖C_k‖² touches only row k), so only the
treated rows change and all three arms are recomputed in one streaming pass over the cached
clean+corrupt embeddings — no new corruption pass. All arms use the MEAN estimator for the
re-embedded centroid; the intervention−control contrast cancels the estimator choice (the frozen
bank uses medoids), so absolute d_k' is compared within-arm, not against the frozen d_k.

Reading: d_k(intervention) >> d_k(control) ≈ d_k(orig)  ⇒ resolution CAUSES fragility (the §3.4.3
robust ranking is partly a crop-statistics artifact — weakens the semantic montage reading,
NOT §3.4.5 pruning, whose mechanism is μ⊥). d_k(intervention) ≈ d_k(control) ⇒ content, not quality.

Phases:
  --phase embed  (GPU) : decode+downsample member crops, CLIP-embed, write centroids_causal.npz
  --phase dk     (CPU) : recompute d_k' for every treated centroid, write causal_dk_table.csv

Consumes membership.npz + fragile_median_min_side from run.py (T1a); run that first.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from cbm_in_the_wild.paths import CHECKPOINT_DIR, REPO_ROOT

HERE       = Path(__file__).resolve().parent
REPO       = HERE.parents[1]
CHECKPOINT = "imagenet_baseline"
DCBM_REPO  = Path(os.environ.get("DCBM_REPO", REPO_ROOT.parent / "DCBM"))
CROPS_ROOT = DCBM_REPO / "data/Segments/imagenet_SAM2/crops"
CROP_BLOB  = DCBM_REPO / "data/Segments/segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch"
SHIFT_DIR  = REPO / "experiments/imagenet_concept_shift_robust_set"
R          = int(os.environ.get("R", "50"))              # concepts per arm
MAX_MEMBERS = int(os.environ.get("MAX_MEMBERS", "200"))  # cap members/concept (bound GPU pass)


def _out_paths(rep_crop: bool) -> tuple[Path, Path, Path]:
    """(centroids.npz, dk_table.csv, summary.json) — suffixed in the n=1 rep-crop mode
    so it never clobbers the cluster-level (§2.2.2) artifacts."""
    if rep_crop:
        return (HERE / "centroids_repcrop.npz", HERE / "causal_dk_table_repcrop.csv",
                HERE / "summary_causal_repcrop.json")
    return (HERE / "centroids_causal.npz", HERE / "causal_dk_table.csv",
            HERE / "summary_causal.json")


def select_concepts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(robust R, mid R, d_k over all K) by the frozen §3.4 ranking."""
    d_k = pd.read_csv(SHIFT_DIR / "concept_shift_table.csv").sort_values("concept_id")["d_k"].values
    order = np.argsort(d_k)
    robust = order[:R]
    mid = order[len(order) // 2 - R // 2: len(order) // 2 - R // 2 + R]
    return robust, mid, d_k


def rep_crop_files() -> tuple[np.ndarray, list, list]:
    """(rep_idx (K,), names, wnids) — rep_idx[k] is the blob index of concept k's
    representative (medoid) crop = the SAM2 crop with max cosine to centroid k. Recomputed
    from the crop blob exactly as run.py:131-140 (rep_idx is not persisted in membership.npz).
    This single crop *is* the frozen bank row C_k up to the medoid tie-break, so the n=1
    intervention degrades precisely the crop the concept was acquired from."""
    from cbm_in_the_wild.dcbm.persistence import load_cbm_artifacts

    blob = torch.load(CROP_BLOB, map_location="cpu", weights_only=False)
    names = list(blob.keys())
    emb   = torch.stack([blob[k][0].float() for k in names])
    wnids = [blob[k][1] for k in names]
    del blob

    C  = load_cbm_artifacts(CHECKPOINT_DIR / CHECKPOINT)["centroids"].float()
    Cn = torch.nn.functional.normalize(C, dim=1)
    K  = C.shape[0]
    rep_sim = torch.full((K,), -float("inf")); rep_idx = torch.zeros(K, dtype=torch.long)
    for s in range(0, len(names), 50_000):
        e   = torch.nn.functional.normalize(emb[s:s + 50_000], dim=1)
        sim = e @ Cn.t()                                   # (chunk, K)
        cmx, carg = sim.max(dim=0)                         # best crop per concept in-chunk
        upd = cmx > rep_sim
        rep_sim[upd] = cmx[upd]; rep_idx[upd] = carg[upd] + s
    return rep_idx.numpy(), names, wnids


def phase_embed(rep_crop: bool = False) -> None:
    import clip
    from PIL import Image

    z = np.load(HERE / "membership.npz", allow_pickle=True)
    names, wnids, assign = z["names"], z["wnids"], z["assign"]
    target = float(z["fragile_median_min_side"])
    robust, mid, _ = select_concepts()
    centroids_f, _, _ = _out_paths(rep_crop)
    mode = "rep-crop n=1 (medoid only)" if rep_crop else f"cluster mean (<={MAX_MEMBERS} members)"
    print(f"target min_side={target:.0f}px  R={R}  estimator={mode}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()
    rng = np.random.default_rng(0)

    def downsample(im: "Image.Image") -> "Image.Image":
        w, h = im.size
        m = min(w, h)
        if m <= target:                                  # already at/below target — can't add info
            return im
        scale = target / m
        return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)

    if rep_crop:
        # n=1: embed ONLY the representative (medoid) crop per concept — the naive single-crop
        # probe §2.2.2 argues against (high-variance, content-confounded). Kept as the motivating
        # baseline so the reader sees the noise floor the cluster estimator was built to beat.
        rep_idx, rnames, rwnids = rep_crop_files()

        def centroid(concept: int, degrade: bool) -> np.ndarray | None:
            i = int(rep_idx[concept])
            try:
                im = Image.open(CROPS_ROOT / str(rwnids[i]) / str(rnames[i])).convert("RGB")
            except (FileNotFoundError, OSError):
                return None
            if degrade:
                im = downsample(im)
            with torch.no_grad():
                e = model.encode_image(preprocess(im).unsqueeze(0).to(device)).cpu().float()
            return e[0].numpy()                           # single-crop embedding (n=1)
    else:
        def centroid(concept: int, degrade: bool) -> np.ndarray | None:
            idx = np.where(assign == concept)[0]
            if len(idx) == 0:
                return None
            if len(idx) > MAX_MEMBERS:
                idx = rng.choice(idx, MAX_MEMBERS, replace=False)
            tens = []
            for i in idx:
                try:
                    im = Image.open(CROPS_ROOT / str(wnids[i]) / str(names[i])).convert("RGB")
                except (FileNotFoundError, OSError):
                    continue
                if degrade:
                    im = downsample(im)
                tens.append(preprocess(im).unsqueeze(0))
            if not tens:
                return None
            with torch.no_grad():
                e = model.encode_image(torch.cat(tens).to(device)).cpu().float()
            return e.mean(0).numpy()                       # MEAN estimator (see module docstring)

    rows = {"concept": [], "arm": [], "centroid": []}
    for arm, concepts, degrade in [("intervention", robust, True),
                                   ("control", robust, False),
                                   ("general", mid, True)]:
        for k in concepts:
            c = centroid(int(k), degrade)
            if c is None:
                continue
            rows["concept"].append(int(k)); rows["arm"].append(arm); rows["centroid"].append(c)
        print(f"  {arm}: {sum(a == arm for a in rows['arm'])} centroids", flush=True)

    np.savez(centroids_f, concept=np.array(rows["concept"]), arm=np.array(rows["arm"], dtype=object),
             centroid=np.stack(rows["centroid"]).astype(np.float32), target_min_side=target)
    print(f"wrote {centroids_f.name} ({len(rows['concept'])} centroids)", flush=True)


def phase_dk(rep_crop: bool = False) -> None:
    from cbm_in_the_wild.dcbm.concept_shift import recompute_dk

    centroids_f, table_f, summary_f = _out_paths(rep_crop)
    z = np.load(centroids_f, allow_pickle=True)
    concept, arm, cents = z["concept"], z["arm"], z["centroid"]
    # d_k is per-row; stack all treated centroids into one bank, recompute in a single pass
    dk_new = recompute_dk([torch.from_numpy(cents)])[0]

    _, _, d_k_orig = select_concepts()
    out = pd.DataFrame({"concept_id": concept, "arm": [str(a) for a in arm],
                        "d_k_orig": d_k_orig[concept], "d_k_new": dk_new})
    out.to_csv(table_f, index=False)

    med = out.groupby("arm").agg(d_k_orig_med=("d_k_orig", "median"),
                                 d_k_new_med=("d_k_new", "median"),
                                 n=("concept_id", "size"))
    summary = {"method": "crop_quality_causal", "estimator": "rep_crop_n1" if rep_crop else "cluster_mean",
               "R": R, "max_members": None if rep_crop else MAX_MEMBERS,
               "target_min_side": float(z["target_min_side"]),
               "by_arm": med.round(4).reset_index().to_dict("records")}
    # headline contrast: intervention rise above the control noise floor
    def m(a): return float(out[out.arm == a]["d_k_new"].median())
    summary["intervention_minus_control_median"] = round(m("intervention") - m("control"), 4)
    summary["control_minus_orig_median"] = round(
        m("control") - float(out[out.arm == "control"]["d_k_orig"].median()), 4)
    summary_f.write_text(json.dumps(summary, indent=2))
    print(med.round(4).to_string(), flush=True)
    print("intervention − control (median d_k'):",
          summary["intervention_minus_control_median"], flush=True)
    print("control − orig (noise floor):", summary["control_minus_orig_median"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["embed", "dk"], required=True)
    ap.add_argument("--rep-crop", action="store_true",
                    help="n=1 probe: degrade/re-embed only the medoid crop (§2.2.1), not the "
                         "cluster mean (§2.2.2). Writes *_repcrop artifacts.")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    (phase_embed if args.phase == "embed" else phase_dk)(rep_crop=args.rep_crop)


if __name__ == "__main__":
    main()
