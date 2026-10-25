#!/usr/bin/env python
"""§3.4.5 capacity-matched validation — native-K baseline vs prune-from-large (CPU-only).

Fixes the capacity confound in the v1 reduced-bank ablation (§3.4.4): that curve varied
BOTH the number of concepts (capacity) AND which concepts were kept (selection), so nothing
could be attributed to robustness-selection. Here capacity is held fixed at each K_wanted and
ONLY the selection differs:

  Arm A — native baseline : C_native = KMeans(K_wanted) on the full clean SAM2 crop pool
                            (`centroids_from_segcrop`, random_state = seed) — a bank built
                            natively at the small size.
  Arm B — prune-from-large: keep the K_wanted most ROBUST concepts of the frozen K_max=2048
                            bank (= rank[:K_wanted]), one arm per ranking in RANK_BY:
                              dk_hold3  (default) — d_k recomputed from shift_grids.npz with
                                        sev-3 EXCLUDED. The sev-3 eval cells then never enter
                                        the selection statistic → no corruption-realization
                                        leakage (the sev-3 ranking is Spearman 0.9997 vs the
                                        all-sev one, so nothing is lost). Image-level overlap
                                        remains (d_k comes from the same 50k test images) —
                                        weak for a K-dim statistic, but state it in the writeup.
                              dk        — the frozen all-severity ranking
                                        (concept_shift_table.csv); kept so the dk_hold3 − dk
                                        delta quantifies the leakage effect.
                              mk_perp   — softmax-invariant residual ranking from
                                        diag_shift_covariance.py v2 (shift_covariance.npz);
                                        sev-3-derived ⇒ exploratory only.
                              rand<i>   — random permutation (rng seed <i>) — the selection-
                                        NULL control. Pruning to K_wanted + refit changes
                                        accuracy regardless of ranking, so a ranking only
                                        "works" if it beats the random band at the same
                                        K_wanted. Gate v2 predicts dk ≈ random (ρ(m_k⊥, d_k)
                                        = 0.089 — d_k is dominated by the per-image uniform
                                        offset softmax cancels).

Both banks have identical dimension K_wanted ⇒ identical capacity ⇒ the delta B − A under
corruption (esp. the corrupt/clean retention ratio) is clean evidence that "build big, prune
fragile" beats "build small natively". Downstream is the shared v1 pipeline: recompute the
reduced-bank activations a = softmax(z·Cᵀ/‖C‖²) over the arm's own bank, refit a linear head
→1000 on the 128k held-out val split, evaluate clean + corrupt (sev-3 mean) on the disjoint
full 50k test. Both arms run over 3 seeds (KMeans random_state + val-subsample) for variance.

K_wanted ∈ {50,100,200,400,800,1200,1600, K_max=2048}; K_max is the convergence anchor where
arm A (native full) ≈ arm B (frozen full) — a sanity check that the setup is unbiased.

GATE OUTCOME (2026-07-06, job 5816007): verdict_v2 = GENERAL — the softmax-visible residual is
low-rank (frac_top1 0.53, PR 3.3) and uncorrelated with d_k (Spearman 0.089). The sweep runs
anyway, REFRAMED as the empirical selection-null test the diagnostic predicts: dk arms should
land inside the rand<i> control band (supervisor's d_k idea closed with accuracy numbers, not
just a diagnostic), and mk_perp measures whether the best-case visible-shift ranking buys
anything at matched capacity.

Native banks are cached to `capacity_matched/native_K{K}_seed{s}.pt` (skip-if-exists → the job
is restartable; the KMeans sweep is the wall-clock driver). Producer for
`ablation_capacity_matched.csv`; §3.4.5 presents A vs B.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cbm_in_the_wild.paths import CHECKPOINT_DIR, REPO_ROOT
from cbm_in_the_wild.dcbm.persistence import load_cbm_artifacts
from cbm_in_the_wild.dcbm.analysis import load_per_image_cell
from cbm_in_the_wild.dcbm.bank import stack_segcrop_emb, centroids_from_segcrop

HERE       = Path(__file__).resolve().parent
REPO       = HERE.parents[1]
CHECKPOINT = "imagenet_baseline"
PROP_ROOT  = REPO / "experiments/imagenet_c_propagation"
DCBM_REPO  = Path(os.environ.get("DCBM_REPO", REPO_ROOT.parent / "DCBM"))
DCBM_EMB   = DCBM_REPO / "data/Embeddings"
VAL_EMB_FILE  = "images_ImageNet_val_CLIP-ViT-B16.torch"                 # 128,118 held-out train imgs
CROP_BLOB     = DCBM_REPO / "data/Segments/segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch"

FAMILIES = ["fog", "gaussian_noise", "motion_blur", "patch_masking", "salt_and_pepper",
            "jpeg_compression", "row_column_dropout", "random_pixel_dropout",
            "sparse_random_occlusion", "structured_pixel_dropout"]
SEV      = 3
SEEDS    = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",") if s.strip()]
# INCLUDE_NATIVE=0 drops the auto A_native arm — lets a shard run one B ranking in isolation
# (native then runs in its own shard) so the grid parallelizes cleanly with no duplicate fits.
INCLUDE_NATIVE = os.environ.get("INCLUDE_NATIVE", "1") == "1"
# 200 left lbfgs unconverged in the v1 runs (ConvergenceWarnings in 5655157/5659433 .err;
# refit full-K clean 0.629–0.663 vs deployed 0.698) — the r-curve shape was partly an
# optimization artifact. 1000 + the per-fit n_iter print below make convergence auditable.
MAX_ITER = int(os.environ.get("MAX_ITER", "1000"))
# arm-B ranking(s), comma-separated:
#   dk_hold3 | dk | mk_perp | rand<i> (selection-null)
#   dk_disjoint — dk (all severities) estimated on a disjoint train pool (env DK_DISJOINT_TABLE)
#   dk_loco_<corr> — leave-one-corruption-out: d_k averaged over the other 9 corruptions
#                (all severities) of the disjoint-pool grid in env DK_LOCO_GRID
#   geom       — T4 geometry arm: medoid-crop min_side, largest first (corruption-free rule;
#                rows are medoids, so each concept has an acquisition resolution)
#   dkres_geom — dk_hold3 OLS-residualized on log2(min_side): d_k with the crop-quality
#                component partialed out (follow-up to imagenet_crop_quality_dk)
RANK_BY  = [s for s in os.environ.get("RANK_BY", "dk_hold3,rand0,rand1,rand2,geom,dkres_geom,geom_ar").split(",") if s]
# KMeans restarts for the native arm — lower than sklearn's default 10 to bound the sweep cost.
N_INIT   = int(os.environ.get("N_INIT", "3"))
# optional subsample of the crop pool for the native KMeans (0 = full pool). A subsample is
# still a legitimate native bank; trims the K=1600/2048 fits. Fixed rng(0) → the same pool
# across seeds; seed variance enters via KMeans random_state + the fit subsample.
CROP_SUBSAMPLE = int(os.environ.get("CROP_SUBSAMPLE", "0"))


def _frozen_medoid_minside(here: Path) -> np.ndarray:
    """min(w,h) of the medoid crop behind each frozen-bank row (cached to npz).

    Rows are medoids (centroid_method='median'), so each row IS one crop's embedding:
    recover the crop key by exact cosine match (bank_crop_keys), then look up its pixel
    dims by filename in the crop-quality experiment's JPEG-header cache.
    """
    cache = here / "geom_frozen2048.npz"
    if cache.exists():
        return np.load(cache)["min_side"]
    from cbm_in_the_wild.dcbm.bank import bank_crop_keys
    art  = load_cbm_artifacts(CHECKPOINT_DIR / CHECKPOINT)
    blob = torch.load(CROP_BLOB, weights_only=False)
    keys = bank_crop_keys(blob, art["centroids"].float())
    del blob
    dims = np.load(REPO / "experiments/imagenet_crop_quality_dk/crop_dims.npz", allow_pickle=True)
    n2ms = dict(zip(dims["names"].tolist(), np.minimum(dims["w"], dims["h"]).tolist()))
    ms   = np.array([n2ms[k] for k in keys], dtype=np.float64)
    np.savez(cache, min_side=ms, crop_keys=np.array(keys, dtype=object))
    print(f"frozen-bank medoid min_side: med={np.nanmedian(ms):.0f}px "
          f"[{np.nanmin(ms):.0f}, {np.nanmax(ms):.0f}], "
          f"{int((~np.isfinite(ms)).sum())} unreadable", flush=True)
    return ms


def _frozen_medoid_ar(here: Path) -> np.ndarray:
    """max/min aspect ratio of the medoid crop behind each frozen-bank row (AR>=1, 1=square).

    Reuses the crop-key alignment cached by _frozen_medoid_minside (geom_frozen2048.npz),
    so the keys are identical to the geom arm — only the metric read from crop_dims differs.
    """
    _frozen_medoid_minside(here)                              # ensure geom_frozen2048.npz exists
    keys = np.load(here / "geom_frozen2048.npz", allow_pickle=True)["crop_keys"].tolist()
    dims = np.load(REPO / "experiments/imagenet_crop_quality_dk/crop_dims.npz", allow_pickle=True)
    w = dict(zip(dims["names"].tolist(), dims["w"].tolist()))
    h = dict(zip(dims["names"].tolist(), dims["h"].tolist()))
    ar = np.array([(max(w[k], h[k]) / min(w[k], h[k])) if (k in w and min(w[k], h[k]) > 0)
                   else np.nan for k in keys], dtype=np.float64)
    print(f"frozen-bank medoid AR: med={np.nanmedian(ar):.2f} "
          f"[{np.nanmin(ar):.2f}, {np.nanmax(ar):.2f}], "
          f"{int((~np.isfinite(ar)).sum())} unreadable", flush=True)
    return ar


def _dk_hold(here: Path, sev: int) -> np.ndarray:
    z    = np.load(here / "shift_grids.npz")
    keep = [i for i, s in enumerate(z["severities"]) if int(s) != sev]
    return np.nanmean(z["dk_grid"][:, :, keep], axis=(1, 2))


def _dk_loco(held: str) -> np.ndarray:
    """d_k over every corruption except `held`, all severities (disjoint-pool grid)."""
    z    = np.load(os.environ["DK_LOCO_GRID"])
    fams = [str(f) for f in z["families"]]
    keep = [i for i, f in enumerate(fams) if f != held]
    assert len(keep) == len(fams) - 1, f"held-out corruption {held!r} not in {fams}"
    return np.nanmean(z["dk_grid"][:, keep, :], axis=(1, 2))


def load_rankings(here: Path, sev: int, k_max: int) -> dict[str, np.ndarray]:
    """concept ids sorted most-robust-first (rand<i>: arbitrary), one entry per RANK_BY name."""
    out: dict[str, np.ndarray] = {}
    for name in RANK_BY:
        if name.startswith("rand"):
            out[name] = np.random.default_rng(int(name.removeprefix("rand"))).permutation(k_max)
        elif name == "dk":
            df = pd.read_csv(here / "concept_shift_table.csv")
            out[name] = df.sort_values("d_k")["concept_id"].values.astype(int)
        elif name == "dk_hold3":
            out[name] = np.argsort(_dk_hold(here, sev))
        elif name == "dk_disjoint":
            # d_k (all severities) re-estimated on a train pool disjoint from the eval images
            # (experiments/imagenet_dk_disjoint_pool/compute_dk.py)
            df = pd.read_csv(os.environ["DK_DISJOINT_TABLE"])
            out[name] = np.argsort(df["d_k"].values)
        elif name.startswith("dk_loco_"):
            out[name] = np.argsort(_dk_loco(name.removeprefix("dk_loco_")))
        elif name == "geom":
            ms = _frozen_medoid_minside(here)
            out[name] = np.argsort(-np.where(np.isfinite(ms), ms, -np.inf), kind="stable")
        elif name == "geom_ar":
            ar = _frozen_medoid_ar(here)                      # square-ish (AR->1) = robust
            out[name] = np.argsort(np.where(np.isfinite(ar), ar, np.inf), kind="stable")
        elif name == "dkres_geom":
            from scipy.stats import spearmanr
            dk = _dk_hold(here, sev)
            ms = _frozen_medoid_minside(here)
            ls = np.log2(np.where(np.isfinite(ms), ms, np.nanmedian(ms)))
            A  = np.c_[np.ones_like(ls), ls]
            coef, *_ = np.linalg.lstsq(A, dk, rcond=None)
            print(f"dkres_geom: spearman(dk_hold3, medoid min_side)="
                  f"{spearmanr(dk, ms, nan_policy='omit')[0]:.4f}, "
                  f"ols slope={coef[1]:.5f}/log2(px)", flush=True)
            out[name] = np.argsort(dk - A @ coef)
        elif name == "mk_perp":
            z = np.load(here / "shift_covariance.npz")
            if "m_k_perp" not in z:
                raise KeyError("shift_covariance.npz has no m_k_perp — re-run "
                               "diag_shift_covariance.py (v2) first")
            out[name] = np.argsort(z["m_k_perp"])
        else:
            raise ValueError(f"unknown RANK_BY entry: {name!r}")
    return out


def main() -> None:
    torch.set_grad_enabled(False)
    # OUT_DIR: where THIS run's results/CSV/summary/figure land (default HERE). NATIVE_CACHE:
    # where native KMeans banks live — fit_n-INDEPENDENT, so a full-val run (OUT_DIR elsewhere)
    # REUSES the original banks instead of rebuilding. Both default to keep original behavior
    # byte-identical (OUT_DIR=HERE, native cache = HERE/capacity_matched).
    OUT_DIR      = Path(os.environ.get("OUT_DIR", str(HERE)))
    native_cache = Path(os.environ.get("NATIVE_CACHE", str(HERE / "capacity_matched")))
    out_cache    = OUT_DIR / "capacity_matched"
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    native_cache.mkdir(parents=True, exist_ok=True)
    out_cache.mkdir(parents=True, exist_ok=True)

    ckpt = CHECKPOINT_DIR / CHECKPOINT
    art  = load_cbm_artifacts(ckpt)
    C    = art["centroids"].float()                      # (K, D) frozen bank
    K, D = C.shape
    # SIZES override (T2 §3.4.5c fine-K fill): comma-separated K_wanted list, sentinel "K"
    # = K_max. Default is the original v2 log-grid. Cells already in results.jsonl are
    # skipped, so `SIZES=400,500,600,700,800` only fits the new interior points; the CSV
    # is rebuilt from the full results.jsonl below, so a partial run augments, never clobbers.
    _sz_env = os.environ.get("SIZES", "")
    SIZES = ([K if s.strip() == "K" else int(s) for s in _sz_env.split(",") if s.strip()]
             if _sz_env else [100, 400, 800, 1200, 1600, K])

    # --- eval side: full 50k clean test + aligned labels (deployed space) ----
    cdir = ckpt / "clean_propagation"
    TEST_EMB = torch.load(cdir / "clip_emb.pt",  map_location="cpu").float()   # (50000, D)
    TEST_LAB = torch.load(cdir / "labels.pt",    map_location="cpu").long().numpy()
    _ids     = torch.load(cdir / "image_ids.pt", map_location="cpu")
    TEST_IDS = _ids.tolist() if isinstance(_ids, torch.Tensor) else list(_ids)
    POS      = {iid: i for i, iid in enumerate(TEST_IDS)}
    Nte      = TEST_EMB.shape[0]

    # --- wnid→index map, derived zero-assumption from (test ids, labels) + DCBM meta ---
    def nk(x: str) -> str:
        x = str(x)
        return x if x.endswith(".JPEG") else x + ".JPEG"

    dtest = torch.load(DCBM_EMB / "images_ImageNet_test_CLIP-ViT-B16.torch", map_location="cpu")
    w2i: dict[str, int] = {}
    for iid, y in zip(TEST_IDS, TEST_LAB):
        w2i[dtest[nk(iid)][1]] = int(y)
    assert sorted(w2i.values()) == list(range(len(w2i))), "wnid→idx not a clean bijection"
    del dtest

    # --- fit side: cached val embeddings (held-out train imgs; disjoint from test) ---
    dval = torch.load(DCBM_EMB / VAL_EMB_FILE, map_location="cpu")
    vkeys = list(dval.keys())
    VAL_EMB = torch.stack([dval[k][0].float() for k in vkeys])          # (Nval, D)
    VAL_LAB = np.array([w2i[dval[k][1]] for k in vkeys], dtype=np.int64)
    del dval
    assert VAL_EMB.shape[1] == D, f"val emb dim {VAL_EMB.shape[1]} != bank D {D}"
    VAL_CLASSES = np.unique(VAL_LAB)

    # optional class-stratified fit subsample (per seed → variance on both arms)
    FIT_N = int(os.environ.get("FIT_N", "0")) or None   # 0 = full 128,118-image pool
    # effective N actually used (full pool when FIT_N is 0/None/>=pool) — stamps every result
    # row so the cache never confuses a 25k fit with a 128k fit for the same (arm, K, seed).
    FIT_N_EFF = int(FIT_N) if (FIT_N and FIT_N < VAL_EMB.shape[0]) else int(VAL_EMB.shape[0])

    def fit_subsample(seed: int):
        if not FIT_N or FIT_N >= VAL_EMB.shape[0]:
            return VAL_EMB, VAL_LAB
        per = max(1, FIT_N // len(VAL_CLASSES))
        rng = np.random.default_rng(seed)
        keep = np.concatenate([
            rng.choice(idx := np.where(VAL_LAB == c)[0], size=min(per, len(idx)), replace=False)
            for c in VAL_CLASSES])
        return VAL_EMB[keep], VAL_LAB[keep]

    # arm-B rankings (most robust first), one arm per RANK_BY entry
    ranks = load_rankings(HERE, SEV, K)
    print("arm-B rankings:", {k: v[:5].tolist() for k, v in ranks.items()}, flush=True)

    # corrupt sev-3 cells (shared across arms/sizes/seeds)
    corrupt_cells = []
    for fam in FAMILIES:
        cell = load_per_image_cell(PROP_ROOT / f"{fam}_s{SEV}" / "per_image", fam, SEV)
        cids = cell["image_ids"]
        cids = cids.tolist() if isinstance(cids, torch.Tensor) else list(cids)
        lab  = TEST_LAB[np.array([POS[i] for i in cids])]
        corrupt_cells.append((cell["clip_emb"].float().contiguous(), lab))
        del cell

    # crop pool (loaded once; only when a native bank actually needs building)
    _pool = {"emb": None}
    def crop_pool() -> np.ndarray:
        if _pool["emb"] is None:
            emb = stack_segcrop_emb(CROP_BLOB)
            if CROP_SUBSAMPLE and CROP_SUBSAMPLE < emb.shape[0]:
                emb = emb[np.random.default_rng(0).choice(emb.shape[0], CROP_SUBSAMPLE, replace=False)]
            _pool["emb"] = emb
            print(f"crop pool: {emb.shape[0]} crops (subsample={CROP_SUBSAMPLE or 'full'})", flush=True)
        return _pool["emb"]

    def native_bank(k_wanted: int, seed: int) -> torch.Tensor:
        f = native_cache / f"native_K{k_wanted}_seed{seed}.pt"
        if f.exists():
            return torch.load(f, map_location="cpu").float()
        Cn = centroids_from_segcrop(crop_pool(), k_wanted, random_state=seed, n_init=N_INIT)
        torch.save(Cn, f)
        print(f"  built native K={k_wanted} seed={seed} -> {tuple(Cn.shape)}", flush=True)
        return Cn

    def acts(emb: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
        raw = (emb @ bank.t()) / (bank ** 2).sum(1)      # (n, k), per-concept /‖C‖²
        return torch.softmax(raw, dim=1).numpy()

    def fit_eval(bank: torch.Tensor, seed: int) -> tuple[float, float, float, int, dict]:
        xe, ye = fit_subsample(seed)
        xfit = acts(xe, bank)
        sc   = StandardScaler().fit(xfit)
        clf  = LogisticRegression(max_iter=MAX_ITER, n_jobs=-1, C=1.0)
        clf.fit(sc.transform(xfit), ye)
        n_it = int(np.max(clf.n_iter_))
        clean = float(clf.score(sc.transform(acts(TEST_EMB, bank)), TEST_LAB))
        cor   = [float(clf.score(sc.transform(acts(e, bank)), lab)) for e, lab in corrupt_cells]
        return clean, float(np.mean(cor)), float(np.std(cor)), n_it, dict(zip(FAMILIES, cor))

    # per-fit result cache: 168 logistic fits at MAX_ITER=1000 can outrun the walltime —
    # completed (arm, K, seed) cells are appended to results.jsonl and skipped on re-submit
    # (afterany chain resumes for free; native banks are cached separately above).
    results_f = out_cache / "results.jsonl"
    done: dict[tuple, dict] = {}
    if results_f.exists():
        for line in results_f.read_text().splitlines():
            rec = json.loads(line)
            done[(rec["arm"], rec["K_wanted"], rec["seed"])] = rec
        print(f"resuming: {len(done)} fits cached in {results_f}", flush=True)

    rows = []
    for r in SIZES:
        for seed in SEEDS:
            arm_defs = ([("A_native", lambda r=r, s=seed: native_bank(r, s))]   # arm A: native KMeans
                        if INCLUDE_NATIVE else [])
            arm_defs += [(f"B_prune_{name}",
                          lambda rk=rk, r=r: C[torch.tensor(rk[:r], dtype=torch.long)])
                         for name, rk in ranks.items()]           # arm B: prune-from-large
            for arm, make_bank in arm_defs:
                key = (arm, int(r), seed)
                cached = done.get(key)
                # fit_n guards the cache: a legacy record (no fit_n) defaults to THIS run's N —
                # safe because legacy cells live only in the original 25k dir whose runs are also
                # 25k. A genuine fit_n mismatch (mixed N in one dir) forces a fresh fit.
                if cached is not None and cached.get("fit_n", FIT_N_EFF) == FIT_N_EFF:
                    rows.append(cached)
                    continue
                cl, co, cs, ni, per_fam = fit_eval(make_bank(), seed)
                conv = "" if ni < MAX_ITER else f"  NOT CONVERGED (n_iter={ni})"
                rec = {"arm": arm, "K_wanted": int(r), "seed": seed, "fit_n": FIT_N_EFF,
                       "clean": cl, "corrupt": co, "corrupt_std": cs, "n_iter": ni,
                       "corrupt_per_family": per_fam}
                rows.append(rec)
                with results_f.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"  K={r:>4} seed={seed} {arm}: clean={cl:.4f} corrupt={co:.4f} ±{cs:.4f}{conv}",
                      flush=True)

    # Rebuild the CSV/figure from the FULL results.jsonl (every (arm, K, seed) ever fit),
    # not just this run's SIZES — a fine-K partial run (T2) must augment the grid, not
    # overwrite it. Dedup by (arm, K_wanted, seed) keeping the last write.
    all_rows = {}
    if results_f.exists():
        for line in results_f.read_text().splitlines():
            r = json.loads(line)
            if r.get("fit_n", FIT_N_EFF) != FIT_N_EFF:        # never mix fit_n into one CSV
                continue
            all_rows[(r["arm"], r["K_wanted"], r["seed"])] = r
    ab = pd.DataFrame(list(all_rows.values()) if all_rows else rows)
    ab = ab.drop(columns=["corrupt_per_family"], errors="ignore")
    ab = ab.sort_values(["arm", "K_wanted", "seed"]).reset_index(drop=True)
    ab.to_csv(OUT_DIR / "ablation_capacity_matched.csv", index=False)

    # --- figure: A vs B, mean ± std over seeds ------------------------------
    # rand<i> arms collapse into one "random" group: its std pools perms × seeds = the null band
    # dk_hold3 stays in the CSV/summary (severity-leakage control, Spearman 0.9997 vs dk) but is
    # dropped from the plot: its curves coincide with dk and only add legend noise
    grp = ab[ab["arm"] != "B_prune_dk_hold3"].copy()
    grp["arm"] = grp["arm"].str.replace(r"^B_prune_rand\d+$", "B_prune_random", regex=True)
    agg = grp.groupby(["arm", "K_wanted"]).agg(
        clean_m=("clean", "mean"), clean_s=("clean", "std"),
        corrupt_m=("corrupt", "mean"), corrupt_s=("corrupt", "std")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    fmts  = ["o-", "s-", "^-", "d-", "v-"]
    style = {a: (fmts[i % len(fmts)],
                 "native-K baseline" if a == "A_native"
                 else f"prune-from-{K} ({a.removeprefix('B_prune_')})")
             for i, a in enumerate(sorted(grp["arm"].unique()))}
    for j, (metric, ttl) in enumerate([("clean", "clean (test)"), ("corrupt", "corrupt (sev-3 mean)")]):
        for arm, (ls, lab) in style.items():
            g = agg[agg["arm"] == arm].sort_values("K_wanted")
            ax[j].errorbar(g["K_wanted"], g[f"{metric}_m"], yerr=g[f"{metric}_s"],
                           fmt=ls, capsize=3, label=lab)
        ax[j].set_xscale("log"); ax[j].set_xlabel("K_wanted (capacity)"); ax[j].set_ylabel("accuracy")
        ax[j].set_title(ttl); ax[j].legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figures/I_capacity_matched.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "method": "capacity_matched_A_native_vs_B_prune",
        "checkpoint": CHECKPOINT, "K_max": int(K), "D": int(D),
        "sizes": [int(s) for s in SIZES], "seeds": SEEDS, "sev": SEV,
        "fit_n": FIT_N_EFF, "eval_n": int(Nte), "max_iter": MAX_ITER,
        "rank_by": RANK_BY, "n_init": N_INIT, "crop_subsample": CROP_SUBSAMPLE,
        "val_emb_file": VAL_EMB_FILE, "crop_blob": str(CROP_BLOB),
        "native_cache": str(native_cache), "out_dir": str(OUT_DIR),
        "table": agg.round(4).to_dict("records"),
    }
    (OUT_DIR / "ablation_capacity_matched_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote ablation_capacity_matched.csv + summary + figure to", OUT_DIR)


if __name__ == "__main__":
    main()
