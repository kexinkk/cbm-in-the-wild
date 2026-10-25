"""Non-destructive 90/10 train/val split for ImageNet ImageFolder data.

Splits the post-extract train_full/ into train/ + val/ at 10% val. Uses
**symlinks**, not copies — ImageNet train_full is ~130 GB and duplicating
would blow through scratch quota and extend the split wall-clock by hours.

Layout at entry:
    data/imagenet/train_full/<wnid>/*.JPEG  (1 281 167, from imagenet_to_imagefolder.py)
    data/imagenet/test/<wnid>/*.JPEG        (50 000, DCBM "test" = ILSVRC2012 val)

Layout produced:
    data/imagenet/train/<wnid>/<stem>.JPEG  (symlinks to train_full, ~1.153M files, 90%)
    data/imagenet/val/<wnid>/<stem>.JPEG    (symlinks to train_full, ~128k files, 10%)

Split ratio fixed at 0.10 per Q2 resolution (2026-04-21): train/internal-val/test
= 86.6/9.6/3.8 (train 1 153 050 / internal val 128 117 / test 50 000 =
ILSVRC2012 val). Paper ratios (90/3/7 over combined pool) were considered
and rejected — see dev_log Q2 row and `01_compute_budgets.md` §2.7 v0.6.
`--ratio` kept as a flag for experimentation but the default is the
committed choice.

Idempotent: symlinks already pointing to the right target are no-ops.

Usage:
    python scripts/data_prep/imagenet_split_val.py
    python scripts/data_prep/imagenet_split_val.py --seed 0 --ratio 0.10
    python scripts/data_prep/imagenet_split_val.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from cbm_in_the_wild.paths import imagenet_dir


def split_wnid(
    src_dir: Path,
    train_out: Path,
    val_out: Path,
    ratio: float,
    rng: random.Random,
    dry_run: bool,
) -> tuple[int, int]:
    files = sorted(src_dir.glob("*.JPEG"))
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * ratio)))
    val_files = files[:n_val]
    train_files = files[n_val:]

    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    for src, dest_dir in [(train_files, train_out), (val_files, val_out)]:
        for f in src:
            dest = dest_dir / f.name
            if dest.is_symlink() or dest.exists():
                continue
            if not dry_run:
                os.symlink(f.resolve(), dest)
    return len(train_files), len(val_files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = imagenet_dir()
    train_full = root / "train_full"
    train_out = root / "train"
    val_out = root / "val"

    if not train_full.is_dir():
        raise FileNotFoundError(
            f"{train_full} not found — run imagenet_to_imagefolder.py first."
        )

    rng = random.Random(args.seed)
    wnid_dirs = sorted(d for d in train_full.iterdir() if d.is_dir())
    print(f"Splitting {len(wnid_dirs)} wnid classes at ratio={args.ratio}, seed={args.seed}")

    tot_tr, tot_va = 0, 0
    for i, wd in enumerate(wnid_dirs):
        nt, nv = split_wnid(
            wd, train_out / wd.name, val_out / wd.name,
            args.ratio, rng, args.dry_run,
        )
        tot_tr += nt
        tot_va += nv
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/1000] wnids done — cum train={tot_tr}, val={tot_va}", flush=True)

    print(f"totals — train: {tot_tr}, val: {tot_va}")

    if not args.dry_run:
        got_tr = sum(1 for _ in train_out.rglob("*.JPEG"))
        got_va = sum(1 for _ in val_out.rglob("*.JPEG"))
        print(f"on disk (symlinks) — train: {got_tr}, val: {got_va}")
        return 0 if (got_tr + got_va) == tot_tr + tot_va else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
