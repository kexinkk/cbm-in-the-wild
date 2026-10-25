"""Non-destructive 90/10 train/val split for CIFAR-10 ImageFolder data.

Upstream DCBM's `create_val.py` deletes source files after copying — a one-way
operation that makes re-splitting painful. This script instead preserves the full train set
as `data/cifar10/train_full/` and produces `data/cifar10/{train,val}/` as
independent copies. Re-splitting is as simple as deleting train/ and val/
and re-running.

Layout expected at entry:
    data/cifar10/train/<class>/<idx>.png    (full 50000, as written by
                                              cifar10_to_imagefolder.py)

Layout produced:
    data/cifar10/train_full/<class>/<idx>.png   (renamed from train/)
    data/cifar10/train/<class>/<idx>.png        (new, 45000 copies, 90%)
    data/cifar10/val/<class>/<idx>.png          (new, 5000 copies, 10%)

Idempotent: re-running once the split exists is a no-op.

Seeded via `--seed` (default 0) so the split is deterministic and checkable
across users.

Usage:
    python scripts/data_prep/cifar10_split_val.py
    python scripts/data_prep/cifar10_split_val.py --seed 0 --ratio 0.1
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from cbm_in_the_wild.paths import cifar10_dir


def split_class(
    src_dir: Path,
    train_out: Path,
    val_out: Path,
    ratio: float,
    rng: random.Random,
) -> tuple[int, int]:
    """Copy files from src_dir into train_out and val_out according to ratio.

    Returns (n_train, n_val) written this call.
    """
    files = sorted(src_dir.glob("*.png"))  # sort for deterministic base order
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * ratio)))
    val_files = files[:n_val]
    train_files = files[n_val:]

    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    for f in val_files:
        dest = val_out / f.name
        if not dest.exists():
            shutil.copy(f, dest)
    for f in train_files:
        dest = train_out / f.name
        if not dest.exists():
            shutil.copy(f, dest)
    return len(train_files), len(val_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.10)
    args = parser.parse_args()

    root = cifar10_dir()
    train_dir = root / "train"
    val_dir = root / "val"
    train_full_dir = root / "train_full"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"{train_dir} not found — run cifar10_to_imagefolder.py first."
        )

    # Step 1: if train_full doesn't exist yet, this is the first split. Rename
    # the just-built train/ → train_full/ as the immutable source of truth.
    if not train_full_dir.exists():
        if val_dir.exists():
            # Unusual state — val/ exists but train_full/ doesn't. Bail rather
            # than guess.
            print(
                f"ERROR: {val_dir} exists but {train_full_dir} does not. "
                "Refusing to guess the prior split state.",
                file=sys.stderr,
            )
            return 2
        print(f"Renaming {train_dir} -> {train_full_dir} (snapshot of full train)")
        train_dir.rename(train_full_dir)

    # Step 2: do the split. rng seeded per-seed, traversal over sorted class
    # names for determinism across filesystems.
    rng = random.Random(args.seed)
    class_dirs = sorted(d for d in train_full_dir.iterdir() if d.is_dir())
    total_train, total_val = 0, 0
    for cd in class_dirs:
        nt, nv = split_class(
            src_dir=cd,
            train_out=train_dir / cd.name,
            val_out=val_dir / cd.name,
            ratio=args.ratio,
            rng=rng,
        )
        total_train += nt
        total_val += nv
        print(f"  {cd.name}: train={nt}, val={nv}")

    print(f"totals — train: {total_train}, val: {total_val}")

    # Step 3: sanity-check counts.
    expected_train = int(50000 * (1 - args.ratio))
    expected_val = 50000 - expected_train
    got_train = sum(1 for _ in train_dir.rglob("*.png"))
    got_val = sum(1 for _ in val_dir.rglob("*.png"))
    print(f"on disk — train: {got_train}/{expected_train}, val: {got_val}/{expected_val}")
    ok = got_train == expected_train and got_val == expected_val
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
