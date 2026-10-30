#!/usr/bin/env python
"""DCBM capacity-matched concept selection with the deployed head recipe.

The concept bank is the ONLY variable. Everything downstream is the
`experiments/imagenet_baseline` recipe (epochs 200, bs 512, lambda_1 1e-4, lr 1e-4,
standard-scaled softmax activations) on the 1.15M train split, run through
`dcbm.bridge.build_cbm`. The upstream cluster/centroid steps are skipped: the bank is
injected directly as `cbm.clustered_concepts` (rows of the frozen reference bank),
which is all `preprocess_data` reads.

Arms (K in config `ks`; 2048 = the shared `full` anchor):
  full          reference 2048 bank unchanged (reproduces the deployed head)
  native<K>     fresh KMeans(K) bank (cached by ablation_capacity_matched.py)
  dk<K>         keep K lowest-d_k (all-severity d_k on the test pairs)
  rand<K>       random K-subset (rng = seed)
  dkd<K>        d_k re-estimated on the disjoint train pool
  loco_<c>_<K>  disjoint-pool d_k over the 9 corruptions other than c
  geom<K>       largest medoid-crop short side first (corruption-free)
  dkres<K>      d_k residualized on log2 of the medoid-crop short side

Phases (skip-if-exists, resumable):
  --phase train --arm <arm> --seed N
      Train one (arm, seed) cell; writes runs/{arm}_seed{N}.pt (head state_dict +
      scaler + keep_ids or native_bank + clean val/test metrics). Set TRAIN_DEADLINE
      (unix time) to stop cleanly before a wall limit; rerun to continue.
  --phase eval        (CPU) score every finished run on the full 50k clean test and
      the 10 severity-3 cells (per-image cells of imagenet_c_propagation); appends
      eval_results.jsonl.
  --phase summarize   (CPU) aggregate to summary.json + printed arm table.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

from cbm_in_the_wild.dcbm.analysis import load_per_image_cell
from cbm_in_the_wild.dcbm.bridge import build_cbm, evaluate_split
from cbm_in_the_wild.dcbm.persistence import LinearProbe, load_cbm_artifacts
from cbm_in_the_wild.paths import CHECKPOINT_DIR
from cbm_in_the_wild.runs import capture_context
from cbm_in_the_wild.seeding import set_all_seeds

SHIFT_DIR = REPO / "experiments/imagenet_concept_shift_robust_set"
DK_TABLE  = SHIFT_DIR / "concept_shift_table.csv"          # all-severity d_k (dk arm)
DKD_TABLE = REPO / "experiments/imagenet_dk_disjoint_pool/dcbm_concept_shift_table.csv"
NATIVE_DIR = SHIFT_DIR / "capacity_matched"               # cached native KMeans banks (native arm)
PROP_ROOT = REPO / "experiments/imagenet_c_propagation"
RUNS_DIR  = HERE / "runs"
EVAL_F    = HERE / "eval_results.jsonl"

DKD_GRID  = REPO / "experiments/imagenet_dk_disjoint_pool/dcbm_shift_grids.npz"  # loco arm
GEOM_NPZ  = SHIFT_DIR / "geom_frozen2048.npz"             # medoid-crop min_side per bank row

# arm-name prefixes, longest-first so `dkres`/`dkd` are matched before `dk`.
ARM_PREFIXES = ("native", "dkres", "dkd", "dk", "geom", "rand")


def _allowed(cfg: dict, arm: str) -> bool:
    """Grid arms, extra arms, and any loco/geom/dkres arm at a grid K."""
    if arm in cfg["arms"] + cfg.get("extra_arms", []):
        return True
    try:
        pre, K = split_arm(arm)
    except ValueError:
        return False
    return (pre.startswith("loco_") or pre in ("geom", "dkres")) and K in cfg["ks"]


def load_config() -> dict:
    with open(HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def split_arm(arm: str) -> tuple[str, int | None]:
    """(prefix, K) — K is None for the size-less `full` anchor."""
    if arm == "full":
        return "full", None
    if arm.startswith("loco_"):                          # loco_<held corruption>_<K>
        held, K = arm.removeprefix("loco_").rsplit("_", 1)
        return f"loco_{held}", int(K)
    for pre in ARM_PREFIXES:
        rest = arm[len(pre):]
        if arm.startswith(pre) and rest.isdigit():
            return pre, int(rest)
    raise ValueError(f"unknown arm: {arm!r}")


def _ranking(kind: str) -> np.ndarray:
    """Reference-bank concept ids sorted most-robust-first, per ranking `kind`.

    dk  — all-severity d_k (concept_shift_table.csv), lowest-shift first.
    dkd — all-severity d_k re-estimated on the disjoint train pool (imagenet_dk_disjoint_pool).
    loco_<c> — disjoint-pool d_k over the 9 corruptions other than c (all severities).
    geom — largest medoid-crop min_side first (corruption-free).
    dkres — all-severity d_k OLS-residualized on log2(min_side).
    """
    if kind == "dk":
        import pandas as pd
        return pd.read_csv(DK_TABLE).sort_values("d_k")["concept_id"].values.astype(int)
    if kind == "dkd":
        import pandas as pd
        return np.argsort(pd.read_csv(DKD_TABLE)["d_k"].values).astype(int)
    if kind.startswith("loco_"):
        # leave-one-corruption-out: disjoint-pool d_k over the other 9 corruptions, all sevs
        held = kind.removeprefix("loco_")
        z = np.load(DKD_GRID)
        keep = [i for i, f in enumerate(z["families"]) if str(f) != held]
        assert len(keep) == len(z["families"]) - 1, f"unknown corruption {held!r}"
        return np.argsort(np.nanmean(z["dk_grid"][:, keep, :], axis=(1, 2))).astype(int)
    if kind in ("geom", "dkres"):
        ms = np.load(GEOM_NPZ, allow_pickle=True)["min_side"]
        if kind == "geom":                               # largest medoid crop first
            return np.argsort(-np.where(np.isfinite(ms), ms, -np.inf), kind="stable").astype(int)
        # all-severity d_k with its OLS dependence on log2(min_side) removed
        import pandas as pd
        t = pd.read_csv(DK_TABLE).sort_values("concept_id")
        dk = t["d_k"].values
        ls = np.log2(np.where(np.isfinite(ms), ms, np.nanmedian(ms)))
        A = np.c_[np.ones_like(ls), ls]
        coef, *_ = np.linalg.lstsq(A, dk, rcond=None)
        return np.argsort(dk - A @ coef).astype(int)
    raise ValueError(f"unknown ranking kind: {kind!r}")


def resolve_bank(arm: str, seed: int, C: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (bank (K,D) float32 ndarray, keep_ids | None).

    keep_ids is row indices into the reference bank C for the subset arms and None for
    `native` (a fresh KMeans bank, not a subset). Native banks are the ones cached by
    ablation_capacity_matched.py (capacity_matched/native_K{K}_seed{s}.pt), reused
    verbatim so bank construction is identical across the refit and retrain protocols.
    """
    pre, K = split_arm(arm)
    k_max = C.shape[0]
    if pre == "full":
        return C.float().numpy(), np.arange(k_max)
    if pre == "native":
        f = NATIVE_DIR / f"native_K{K}_seed{seed}.pt"
        if not f.exists():
            raise FileNotFoundError(f"native bank cache missing: {f}")
        b = torch.load(f, map_location="cpu")
        bank = (b.float().numpy() if torch.is_tensor(b) else np.asarray(b, dtype=np.float32))
        if bank.shape[0] != K:
            raise ValueError(f"{f}: expected K={K}, got {bank.shape[0]}")
        return bank, None
    if pre == "rand":
        ids = np.random.default_rng(seed).permutation(k_max)[:K]
    else:                                                # dk / dkd / loco / geom / dkres
        ids = _ranking(pre)[:K]
    ids = np.asarray(ids, dtype=np.int64)
    return C[torch.as_tensor(ids, dtype=torch.long)].float().numpy(), ids


# --------------------------------------------------------------------------- train
def _train_resumable(cbm, cfg: dict, seed: int, ckpt: Path, deadline: float | None):
    """Byte-faithful replication of upstream CBM.train (utils/dcbm.py:884), plus
    per-epoch checkpointing so time-limited segments can chain.

    Kept identical: Adam(lr) on a fresh LinearProbe, loss = CE(one-hot log-softmax)
    + lambda_1·‖W‖₁, DataLoader(shuffle=True) driven by the global torch CPU RNG
    (state saved/restored → batch order matches an uninterrupted run), per-epoch val
    evaluation via upstream cbm.evaluate, best-val-loss weight selection at the end.
    Only deviation: the upstream per-epoch TEST evaluation is skipped — it feeds no
    decision (only wandb logging, off here), so the trained model is unchanged.
    """
    import torch.nn.functional as F
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset

    device, bs = cfg["device"], cfg["batch_size"]
    num_epochs, lambda_1, lr = cfg["epochs"], cfg["lambda_1"], cfg["lr"]
    one_hot = cfg.get("one_hot", False)
    # Upstream's flag is inverted: one_hot=False means labels ARE one-hot floats
    # (cross_entropy_one_hot path) — the only branch replicated here.
    assert not one_hot, "_train_resumable replicates only the one_hot=False recipe branch"

    def cross_entropy_one_hot(inp, target):              # upstream utils/dcbm.py:912
        return -torch.sum(target * F.log_softmax(inp, dim=1), dim=1).mean()

    X_train = torch.tensor(cbm.X_train, dtype=torch.float32).to(device)
    X_val   = torch.tensor(cbm.X_val,   dtype=torch.float32).to(device)
    y_train = torch.tensor(cbm.y_train, dtype=torch.float32).to(device)   # one_hot=False -> one-hot floats
    y_val   = torch.tensor(cbm.y_val,   dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=bs, shuffle=True)

    set_all_seeds(seed)                                  # deterministic LinearProbe init
    model = LinearProbe(cbm.X_train.shape[1], cbm.y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cbm.criterion = cross_entropy_one_hot               # cbm.evaluate reads self.criterion

    start_epoch, best_val_loss = 0, float("inf")
    best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state"]); model.to(device)
        optimizer.load_state_dict(st["opt_state"])
        best_val_loss, best_model_weights = st["best_val_loss"], st["best_model_weights"]
        torch.set_rng_state(st["torch_rng_state"])
        start_epoch = st["epoch_done"]
        print(f"resuming from epoch {start_epoch}/{num_epochs} (best_val_loss {best_val_loss:.4f})")

    for epoch in range(start_epoch, num_epochs):
        te = time.time()
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = cross_entropy_one_hot(outputs, batch_y) \
                 + lambda_1 * torch.norm(model.linear.weight, p=1)
            loss.backward()
            optimizer.step()

        cbm.model = model
        val_loss, val_acc = cbm.evaluate(val_loader, device=device, one_hot=one_hot)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}

        tmp = ckpt.with_suffix(".tmp")
        torch.save({"epoch_done": epoch + 1,
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "opt_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_model_weights": {k: v.cpu() for k, v in best_model_weights.items()},
                    "torch_rng_state": torch.get_rng_state()}, tmp)
        tmp.replace(ckpt)

        dt = time.time() - te
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"epoch {epoch + 1}/{num_epochs}: {dt:.1f}s  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.2f}%", flush=True)
        if deadline is not None and time.time() + 2 * dt > deadline:
            print(f"deadline approaching after epoch {epoch + 1} — exiting for resume "
                  f"(next segment continues from the checkpoint)", flush=True)
            return None

    model.load_state_dict(best_model_weights)            # upstream: restore best-val weights
    cbm.model = model
    return model


def run_train(cfg: dict, arm: str, seed: int) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{arm}_seed{seed}.pt"
    if out.exists():
        print(f"skip: {out} exists")
        return
    deadline = float(os.environ["TRAIN_DEADLINE"]) if os.environ.get("TRAIN_DEADLINE") else None

    t0  = time.time()
    art = load_cbm_artifacts(CHECKPOINT_DIR / cfg["checkpoint"])
    C   = art["centroids"].float()                       # (K, D) frozen reference bank
    bank, ids = resolve_bank(arm, seed, C)   # ids is None for native banks
    src = "native KMeans cache" if ids is None else f"reference {tuple(C.shape)}"
    print(f"arm={arm} seed={seed}: bank {bank.shape} from {src}")

    cbm = build_cbm(cfg=cfg, seed=seed)                  # __init__ loads all three splits
    print(f"embeddings loaded + extracted in {time.time() - t0:.0f}s", flush=True)
    # Inject the bank; skip load_concepts / cluster / centroid entirely.
    cbm.clustered_concepts = bank
    cbm.centroid_method = f"pruned-ref[{arm}]"
    cbm.preprocess_data(type_="standard", label_type=cfg["one_hot"])
    t_pre = time.time()
    print(f"preprocess (softmax acts + scaler) in {t_pre - t0:.0f}s total setup", flush=True)

    model = _train_resumable(cbm, cfg, seed, RUNS_DIR / f"{arm}_seed{seed}.train_ckpt.pt",
                             deadline)
    if model is None:                                    # deadline hit — segment done, not the run
        return
    t_train = time.time()

    val  = evaluate_split(cbm, cbm.X_val,  cbm.y_val,  device=cfg["device"],
                          batch_size=cfg["batch_size"], one_hot=cfg["one_hot"])
    test = evaluate_split(cbm, cbm.X_test, cbm.y_test, device=cfg["device"],
                          batch_size=cfg["batch_size"], one_hot=cfg["one_hot"])
    rec = {
        "arm": arm, "seed": seed,
        # subset arms store row indices into the reference bank; native arms store the
        # bank itself (not reconstructible from the reference) so eval is self-contained.
        "keep_ids": None if ids is None else np.asarray(ids, dtype=np.int64),
        "native_bank": None if ids is not None else np.asarray(bank, dtype=np.float32),
        "head_state": {k: v.cpu() for k, v in cbm.model.state_dict().items()},
        "scaler": cbm.scaler,
        "clean_val": val["accuracy"], "clean_test": test["accuracy"],
        "config": {k: cfg[k] for k in ("epochs", "batch_size", "lambda_1", "lr", "checkpoint")},
        "context": capture_context(REPO),
        "timing_seconds": {"preprocess": t_pre - t0, "train": t_train - t_pre,
                           "total": time.time() - t0},
    }
    torch.save(rec, out)
    (RUNS_DIR / f"{arm}_seed{seed}.train_ckpt.pt").unlink(missing_ok=True)
    print(f"done: {out}  clean_val={val['accuracy']:.4f}  clean_test={test['accuracy']:.4f}  "
          f"({rec['timing_seconds']['train']:.0f}s train this segment)")


# ---------------------------------------------------------------------------- eval
def _acts(emb: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
    """Deployed forward: softmax_K( z·Cᵀ/‖C‖² ) — mirrors upstream preprocess_module."""
    raw = (emb @ bank.t()) / (bank ** 2).sum(1)
    return torch.softmax(raw, dim=1).numpy()


def run_eval(cfg: dict) -> None:
    torch.set_grad_enabled(False)
    done = set()
    if EVAL_F.exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, EVAL_F.read_text().splitlines())}
    pending = [f for f in sorted(RUNS_DIR.glob("*_seed*.pt"))
               if not f.name.endswith(".train_ckpt.pt")
               and (f.stem.rsplit("_seed", 1)[0], int(f.stem.rsplit("_seed", 1)[1])) not in done]
    if not pending:
        print("no finished runs pending evaluation — nothing to do")
        return

    art = load_cbm_artifacts(CHECKPOINT_DIR / cfg["checkpoint"])
    C   = art["centroids"].float()

    cdir     = CHECKPOINT_DIR / cfg["checkpoint"] / "clean_propagation"
    TEST_EMB = torch.load(cdir / "clip_emb.pt",  map_location="cpu").float()
    TEST_LAB = torch.load(cdir / "labels.pt",    map_location="cpu").long().numpy()
    _ids     = torch.load(cdir / "image_ids.pt", map_location="cpu")
    TEST_IDS = _ids.tolist() if isinstance(_ids, torch.Tensor) else list(_ids)
    POS      = {iid: i for i, iid in enumerate(TEST_IDS)}

    cells = []
    for fam in cfg["families"]:
        cell = load_per_image_cell(PROP_ROOT / f"{fam}_s{cfg['sev']}" / "per_image",
                                   fam, cfg["sev"])
        cids = cell["image_ids"]
        cids = cids.tolist() if isinstance(cids, torch.Tensor) else list(cids)
        lab  = TEST_LAB[np.array([POS[i] for i in cids])]
        cells.append((fam, cell["clip_emb"].float().contiguous(), lab))
        del cell
    print(f"eval side ready: clean {tuple(TEST_EMB.shape)} + {len(cells)} sev-{cfg['sev']} cells")

    for f in pending:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        if rec.get("keep_ids") is not None:                          # subset arms
            bank = C[torch.as_tensor(rec["keep_ids"], dtype=torch.long)]
        else:                                                        # native arms
            bank = torch.as_tensor(np.asarray(rec["native_bank"]), dtype=torch.float32)
        head = LinearProbe(bank.shape[0], rec["head_state"]["linear.weight"].shape[0])
        head.load_state_dict(rec["head_state"]); head.eval()
        sc = rec["scaler"]

        def score(emb: torch.Tensor, lab: np.ndarray) -> float:
            x = torch.tensor(sc.transform(_acts(emb, bank)), dtype=torch.float32)
            return float((head(x).argmax(1).numpy() == lab).mean())

        clean = score(TEST_EMB, TEST_LAB)
        # harness check: propagation-cache clean must match the train-phase test acc
        drift = abs(clean - rec["clean_test"])
        assert drift < 0.01, (f"{f.name}: clean drift {drift:.4f} "
                              f"(eval {clean:.4f} vs train {rec['clean_test']:.4f}) — "
                              "label/embedding indexing mismatch")
        cor = {fam: score(emb, lab) for fam, emb, lab in cells}
        row = {"arm": rec["arm"], "seed": rec["seed"], "K": int(bank.shape[0]),
               "clean": clean, "corrupt_mean": float(np.mean(list(cor.values()))),
               "corrupt_std_across_fams": float(np.std(list(cor.values()))),
               "per_family": cor}
        with EVAL_F.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{rec['arm']}_seed{rec['seed']}: clean {clean:.4f}  "
              f"corrupt(sev-{cfg['sev']}) {row['corrupt_mean']:.4f}")


# ----------------------------------------------------------------------- summarize
def run_summarize(cfg: dict) -> None:
    import pandas as pd
    if not EVAL_F.exists():
        print("nothing to summarize yet — no eval_results.jsonl (no finished runs scored)")
        return
    rows = [json.loads(l) for l in EVAL_F.read_text().splitlines()]
    df   = pd.DataFrame(rows)
    agg  = (df.groupby("arm")
              .agg(K=("K", "first"), n=("seed", "count"),
                   clean_m=("clean", "mean"), clean_s=("clean", "std"),
                   corrupt_m=("corrupt_mean", "mean"), corrupt_s=("corrupt_mean", "std"))
              .reset_index().sort_values("K"))
    print(agg.round(4).to_string(index=False))
    expected = len(cfg["arms"]) * len(cfg["seeds"])
    summary = {
        "method": "deployed_recipe_retrain_on_pruned_bank",
        "recipe": {k: cfg[k] for k in ("epochs", "batch_size", "lambda_1", "lr")},
        "checkpoint": cfg["checkpoint"], "sev": cfg["sev"],
        "n_runs": len(rows), "n_expected": expected,
        "complete": len(rows) == expected,
        "stock_reference": {
            "clean_50k": 0.6977,       # deployed head (experiments/imagenet_baseline)
            "sev3_mean_50k": 0.4513,   # deployed head, mean over the 10 severity-3 cells
        },
        "table": agg.round(6).to_dict("records"),
        "context": capture_context(REPO),
    }
    with open(HERE / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {HERE / 'summary.json'}  (complete={summary['complete']}: "
          f"{len(rows)}/{expected} runs)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["train", "eval", "summarize"])
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    cfg = load_config()
    if a.phase == "train":
        if not _allowed(cfg, a.arm) or a.seed not in cfg["seeds"]:
            raise SystemExit(f"--arm must be one of {cfg['arms'] + cfg['extra_arms']} (or a "
                             f"loco_<c>_K / geom K / dkres K arm, K in {cfg['ks']}), "
                             f"--seed one of {cfg['seeds']}")
        run_train(cfg, a.arm, a.seed)
    elif a.phase == "eval":
        run_eval(cfg)
    else:
        run_summarize(cfg)


if __name__ == "__main__":
    main()
