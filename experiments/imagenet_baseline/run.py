"""ImageNet-1k + ViT-B/16 DCBM baseline: sweep, pick best-val, 5-seed replicate.

Structure mirrors `experiments/cifar10_baseline/run.py` exactly;
only the wandb project strings are parametrized on `cfg["dataset"]` so the
same control flow works for both runs. Keep these two files in sync if a
real bug turns up in one — the only *intentional* divergence is the config.

Split into two phases so the sweep can be sharded across a SLURM array
(one cluster count per shard, 3 (λ, lr) cells):

  --phase sweep [--cluster-index N]
      Run the sweep grid. With `--cluster-index N`, run only the 3
      (λ, lr) cells for the N-th cluster count (N ∈ 0..len(cluster list)−1)
      and write `sweep_shard_<N>.jsonl`. Without it, run all clusters
      and write `sweep.jsonl`.

  --phase replicate
      Load all `sweep_shard_*.jsonl` (or `sweep.jsonl`), pick the cell with
      highest val accuracy, replicate over 5 seeds, write `summary.json`
      with pass/fail verdict vs `target_mean_test_accuracy`.

  --phase summarize
      Skip training, write summary.json from existing shard/seed logs.
      (Matches the CIFAR-10 recovery mode added after Apr 19's seed-5
      timeout — safety net for long-running ImageNet replicates.)

  --phase all
      sweep (unsharded) + replicate in one process. Useful for debugging,
      not recommended on the cluster (too long for any short partition).

Typical SLURM submission:
    sbatch scripts/slurm/imagenet1000/06_baseline_sweep.sbatch     # array 0..4
    sbatch scripts/slurm/imagenet1000/07_baseline_replicate.sbatch # afterok
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

from cbm_in_the_wild.dcbm_bridge import build_cbm, evaluate_split
from cbm_in_the_wild.runs import capture_context, write_summary
from cbm_in_the_wild.seeding import set_all_seeds


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def train_and_evaluate(
    cfg: dict,
    *,
    seed: int,
    n_clusters: int,
    lambda_1: float,
    lr: float,
    project: str,
    return_cbm: bool = False,
) -> dict:
    """One end-to-end training run. Returns val + test metrics + timings.

    If ``return_cbm=True`` the trained ``CBM`` instance is included under
    the ``"cbm"`` key — used by ``run_save_best`` to persist artifacts.
    """
    t0 = time.time()
    cbm = build_cbm(cfg=cfg, seed=seed, dcbm_repo=DCBM_REPO)

    segment_path = str(DCBM_REPO / "data" / "Segments") + "/"
    subsets_path = str(DCBM_REPO / "data" / "Embeddings" / "subsets")

    cbm.load_concepts(
        segment_path=segment_path,
        segmentation_technique=cfg["segmentation_technique"],
        concept_name=cfg["concept_name"],
        selected_image_concepts=subsets_path,
        concept_per_class=cfg["concept_per_class"],
        crop=cfg["crop"],
    )
    cbm.cluster_image_concepts(
        clustering_method=cfg["cluster_method"],
        n_clusters=n_clusters,
    )
    cbm.centroid_concepts(centroid_method=cfg["centroid_method"])
    cbm.preprocess_data(type_="standard", label_type=cfg["one_hot"])
    t_pre = time.time()

    # Re-seed so the LinearProbe init is deterministic per-seed, per-config.
    set_all_seeds(seed)
    cbm.train(
        num_epochs=cfg["epochs"],
        lambda_1=lambda_1,
        lr=lr,
        device=cfg["device"],
        batch_size=cfg["batch_size"],
        project=project,
        to_print=False,
        early_stopping_patience=cfg["early_stopping_patience"],
        one_hot=cfg["one_hot"],
        use_wandb=cfg["use_wandb"],
    )
    t_train = time.time()

    val = evaluate_split(
        cbm, cbm.X_val, cbm.y_val,
        device=cfg["device"], batch_size=cfg["batch_size"],
        one_hot=cfg["one_hot"],
    )
    test = evaluate_split(
        cbm, cbm.X_test, cbm.y_test,
        device=cfg["device"], batch_size=cfg["batch_size"],
        one_hot=cfg["one_hot"],
    )
    out = {
        "val": val,
        "test": test,
        "timing_seconds": {
            "preprocess": t_pre - t0,
            "train": t_train - t_pre,
            "total": time.time() - t0,
        },
    }
    if return_cbm:
        out["cbm"] = cbm
    return out


def run_save_best(
    cfg: dict,
    *,
    best: dict,
    out_dir: Path,
    seed: int,
    project: str,
) -> dict:
    """Train one CBM with ``best`` (n_clusters, λ, lr); save artifacts.

    Output layout (per ``cbm_in_the_wild.persistence.save_cbm_artifacts``):
        out_dir/linear_head.pt
        out_dir/centroids.pt
        out_dir/scaler.pkl
        out_dir/manifest.json
    """
    from cbm_in_the_wild.persistence import save_cbm_artifacts

    res = train_and_evaluate(
        cfg, seed=seed,
        n_clusters=best["n_clusters"],
        lambda_1=best["lambda_1"],
        lr=best["lr"],
        project=project,
        return_cbm=True,
    )
    save_cbm_artifacts(
        res["cbm"], out_dir,
        manifest={
            "n_clusters": best["n_clusters"],
            "lambda_1": best["lambda_1"],
            "lr": best["lr"],
            "seed": int(seed),
            "test_accuracy": res["test"]["accuracy"],
            "val_accuracy": res["val"]["accuracy"],
            "train_seconds": res["timing_seconds"]["total"],
            "model_name": cfg["model_name"],
        },
    )
    print(f"[save_best] artifacts → {out_dir} "
          f"(test_acc={res['test']['accuracy']:.4f})", flush=True)
    return res


def run_sweep_cells(
    cfg: dict,
    *,
    cluster_counts: list[int],
    seed: int,
    out_log: Path,
    project: str,
) -> list[dict]:
    """Run 3 (λ, lr) cells for each cluster count in `cluster_counts`."""
    if out_log.exists():
        out_log.unlink()
    rows: list[dict] = []
    for n_clusters in cluster_counts:
        for lambda_1, lr in cfg["sweep"]["lambda_lr_pairs"]:
            tag = f"n={n_clusters} λ={lambda_1} lr={lr}"
            print(f"[sweep] {tag}", flush=True)
            try:
                res = train_and_evaluate(
                    cfg, seed=seed,
                    n_clusters=n_clusters, lambda_1=lambda_1, lr=lr,
                    project=project,
                )
                row = {
                    "phase": "sweep", "seed": seed,
                    "n_clusters": n_clusters, "lambda_1": lambda_1, "lr": lr,
                    "val": res["val"], "test": res["test"],
                    "timing_seconds": res["timing_seconds"],
                }
                print(f"[sweep] {tag} → val={res['val']['accuracy']:.4f} "
                      f"test={res['test']['accuracy']:.4f} "
                      f"({res['timing_seconds']['total']:.1f}s)",
                      flush=True)
            except Exception as e:  # noqa: BLE001 — record and continue
                row = {
                    "phase": "sweep", "seed": seed,
                    "n_clusters": n_clusters, "lambda_1": lambda_1, "lr": lr,
                    "error": f"{type(e).__name__}: {e}",
                }
                print(f"[sweep] {tag} FAILED: {e}", flush=True)
            rows.append(row)
            append_jsonl(out_log, row)
    return rows


def load_all_sweep_rows() -> list[dict]:
    """Merge `sweep.jsonl` + all `sweep_shard_*.jsonl` present."""
    rows: list[dict] = []
    unsharded = HERE / "sweep.jsonl"
    rows += read_jsonl(unsharded)
    for p in sorted(HERE.glob("sweep_shard_*.jsonl")):
        rows += read_jsonl(p)
    return rows


def run_replication(
    cfg: dict,
    *,
    best: dict,
    out_log: Path,
    project: str,
) -> list[dict]:
    if out_log.exists():
        out_log.unlink()
    rows: list[dict] = []
    for seed in cfg["replication_seeds"]:
        print(f"[seed={seed}] training…", flush=True)
        res = train_and_evaluate(
            cfg, seed=int(seed),
            n_clusters=best["n_clusters"],
            lambda_1=best["lambda_1"],
            lr=best["lr"],
            project=project,
        )
        row = {
            "phase": "replication", "seed": int(seed),
            "n_clusters": best["n_clusters"],
            "lambda_1": best["lambda_1"], "lr": best["lr"],
            "val": res["val"], "test": res["test"],
            "timing_seconds": res["timing_seconds"],
        }
        print(f"[seed={seed}] val={res['val']['accuracy']:.4f} "
              f"test={res['test']['accuracy']:.4f}", flush=True)
        rows.append(row)
        append_jsonl(out_log, row)
    return rows


def write_final_summary(
    cfg: dict,
    *,
    context: dict,
    sweep_rows: list[dict],
    seed_rows: list[dict],
    best: dict,
    t_run_start: float,
    target: float,
) -> bool:
    test_accs = [r["test"]["accuracy"] for r in seed_rows]
    val_accs = [r["val"]["accuracy"] for r in seed_rows]
    mean_test = statistics.mean(test_accs)
    std_test = statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0
    mean_val = statistics.mean(val_accs)
    std_val = statistics.stdev(val_accs) if len(val_accs) > 1 else 0.0
    passed = mean_test >= target

    print("=== verdict ===")
    print(f"  target: mean_test ≥ {target:.3f}")
    print(f"  mean_test = {mean_test:.4f} ± {std_test:.4f}")
    print(f"  mean_val  = {mean_val:.4f} ± {std_val:.4f}")
    print(f"  passed    = {passed}")

    write_summary(
        HERE / "summary.json",
        config=cfg,
        context=context,
        timing_seconds={"total": time.time() - t_run_start},
        metrics={
            "best_sweep_cell": {
                "n_clusters": best["n_clusters"],
                "lambda_1": best["lambda_1"],
                "lr": best["lr"],
                "val_accuracy": best["val"]["accuracy"],
                "test_accuracy": best["test"]["accuracy"],
            },
            "replication": {
                "seeds": cfg["replication_seeds"],
                "test_accuracies": test_accs,
                "val_accuracies": val_accs,
                "mean_test_accuracy": mean_test,
                "std_test_accuracy": std_test,
                "mean_val_accuracy": mean_val,
                "std_val_accuracy": std_val,
            },
        },
        extra={
            "target_mean_test_accuracy": target,
            "passed": passed,
            "sweep_rows": sweep_rows,
            "seed_rows": seed_rows,
        },
    )
    print(f"[summary] wrote {HERE / 'summary.json'}")
    return passed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE / "config.yaml")
    p.add_argument(
        "--phase",
        choices=["sweep", "replicate", "summarize", "save_best", "all"],
        default="all",
    )
    p.add_argument(
        "--cluster-index", type=int, default=None,
        help="For --phase sweep only: run just the N-th cluster count "
             "(0-based) and write sweep_shard_<N>.jsonl.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_seed = int(cfg["replication_seeds"][0])
    target = float(cfg["target_mean_test_accuracy"])

    # wandb project names derive from the dataset — no hardcoded cifar10 string.
    sweep_project = f"{cfg['dataset']}-baseline-sweep"
    replicate_project = f"{cfg['dataset']}-baseline-replicate"

    context = capture_context(REPO_ROOT)
    context["dcbm_git_commit"] = capture_context(DCBM_REPO)["git_commit"]

    t_run_start = time.time()

    # ------------- sweep phase -------------
    if args.phase in ("sweep", "all"):
        cluster_counts = cfg["sweep"]["n_clusters"]
        if args.cluster_index is not None:
            if not 0 <= args.cluster_index < len(cluster_counts):
                raise SystemExit(
                    f"--cluster-index {args.cluster_index} out of range "
                    f"[0, {len(cluster_counts)})"
                )
            shard_clusters = [cluster_counts[args.cluster_index]]
            out_log = HERE / f"sweep_shard_{args.cluster_index}.jsonl"
            print(f"=== sweep shard {args.cluster_index}: "
                  f"clusters={shard_clusters}, seed={sweep_seed} ===",
                  flush=True)
        else:
            shard_clusters = cluster_counts
            out_log = HERE / "sweep.jsonl"
            print(f"=== sweep (all clusters, seed={sweep_seed}) ===",
                  flush=True)

        run_sweep_cells(
            cfg,
            cluster_counts=shard_clusters,
            seed=sweep_seed,
            out_log=out_log,
            project=sweep_project,
        )

        if args.phase == "sweep":
            return 0

    # ------------- replicate / summarize phase -------------
    sweep_rows = load_all_sweep_rows()
    valid = [r for r in sweep_rows if "error" not in r]
    if not valid:
        print("[abort] no valid sweep rows found "
              "(checked sweep.jsonl + sweep_shard_*.jsonl)")
        return 1

    best = max(valid, key=lambda r: r["val"]["accuracy"])
    print(f"=== best-val sweep cell ===")
    print(f"  n_clusters={best['n_clusters']} λ={best['lambda_1']} "
          f"lr={best['lr']} val={best['val']['accuracy']:.4f} "
          f"test={best['test']['accuracy']:.4f}")

    if args.phase == "save_best":
        out_dir = REPO_ROOT / "checkpoints" / f"{cfg['dataset']}_baseline"
        run_save_best(
            cfg, best=best, out_dir=out_dir, seed=sweep_seed,
            project=f"{cfg['dataset']}-baseline-save-best",
        )
        return 0

    if args.phase == "summarize":
        seeds_path = HERE / "seeds.jsonl"
        if not seeds_path.exists():
            print(f"[abort] {seeds_path} not found — nothing to summarize")
            return 1
        seed_rows = []
        with open(seeds_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    seed_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[summarize] skipping malformed line {i} "
                          f"in seeds.jsonl (runner probably died mid-write)")
        if not seed_rows:
            print("[abort] seeds.jsonl had no valid rows")
            return 1
        print(f"=== summarize from {len(seed_rows)} existing seed row(s) ===")
    else:
        seed_rows = run_replication(
            cfg, best=best, out_log=HERE / "seeds.jsonl",
            project=replicate_project,
        )

    passed = write_final_summary(
        cfg,
        context=context,
        sweep_rows=sweep_rows,
        seed_rows=seed_rows,
        best=best,
        t_run_start=t_run_start,
        target=target,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
