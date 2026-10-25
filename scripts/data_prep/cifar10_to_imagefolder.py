"""Convert pickled CIFAR-10 batches to DCBM's ImageFolder layout.

Input (read-only, never touched):
    data/raw/cifar-10-batches-py/{data_batch_1..5,test_batch,batches.meta}

Output:
    data/cifar10/train/<class_name>/<global_idx>.png   (50000 files)
    data/cifar10/test/<class_name>/<global_idx>.png    (10000 files)

Idempotent: if a PNG already exists, it is skipped. This lets the script
resume after an interruption without duplicating work. The class-name list
matches what upstream DCBM hardcodes at DCBM/utils/dcbm.py:134.

Usage (activate env first):
    python scripts/data_prep/cifar10_to_imagefolder.py
    python scripts/data_prep/cifar10_to_imagefolder.py --dry-run    # count only
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from cbm_in_the_wild.paths import cifar10_dir, DATA_DIR


RAW_DIR = DATA_DIR / "raw" / "cifar-10-batches-py"

# Paper + dcbm.py:134 label order. Do NOT reorder — the integer labels in the
# pickle batches index into this list, and downstream cached embeddings refer
# to class names by index.
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_batch(path: Path) -> tuple[np.ndarray, list[int]]:
    """Return (images [N, 32, 32, 3] uint8, labels [N])."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # CIFAR-10 pickle stores images as (N, 3072) with channel-first flattened.
    flat = d[b"data"]  # shape (N, 3072) uint8
    imgs = flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # → (N, 32, 32, 3)
    labels = list(d[b"labels"])
    return imgs, labels


def write_split(
    batch_paths: list[Path],
    out_dir: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """Write images to out_dir/<class>/<global_idx>.png. Returns (written, skipped)."""
    for cls in CLASS_NAMES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    global_idx = 0
    for bp in batch_paths:
        imgs, labels = load_batch(bp)
        for img, label in zip(imgs, labels):
            cls = CLASS_NAMES[label]
            dest = out_dir / cls / f"{global_idx}.png"
            if dest.exists():
                skipped += 1
            else:
                if not dry_run:
                    Image.fromarray(img).save(dest)
                written += 1
            global_idx += 1
        print(f"  {bp.name}: {len(imgs)} images cumulatively", flush=True)
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 raw dir not found: {RAW_DIR}. "
            "Expected pickled batches (data_batch_1..5, test_batch)."
        )

    out_root = cifar10_dir()
    print(f"Writing to: {out_root} ({'dry-run' if args.dry_run else 'live'})")

    train_batches = sorted(RAW_DIR.glob("data_batch_*"))
    if len(train_batches) != 5:
        raise RuntimeError(f"Expected 5 train batches, found {len(train_batches)}")
    test_batch = RAW_DIR / "test_batch"
    if not test_batch.exists():
        raise RuntimeError(f"Missing test batch: {test_batch}")

    print("=== train ===")
    w, s = write_split(train_batches, out_root / "train", args.dry_run)
    print(f"train: wrote {w}, skipped {s}")

    print("=== test ===")
    wt, st = write_split([test_batch], out_root / "test", args.dry_run)
    print(f"test: wrote {wt}, skipped {st}")

    # Expected counts: 50000 train + 10000 test, 6000/class + 1000/class.
    expected = {"train": 50000, "test": 10000}
    for split, n_expected in expected.items():
        n_actual = sum(1 for _ in (out_root / split).rglob("*.png"))
        status = "OK" if n_actual == n_expected else "MISMATCH"
        print(f"{split}: {n_actual}/{n_expected} PNGs on disk — {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
