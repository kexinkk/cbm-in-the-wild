"""Build nested 5/10/25/50-per-class concept subsets for ImageNet.

Input:
    data/imagenet/train/<wnid>/<stem>.JPEG   (from imagenet_split_val.py)

Output:
    $DCBM_REPO/data/Embeddings/subsets/ImageNet/ImageNet_rand_{5,10,25,50}.txt

Each line is one JPEG basename (no path, no dir). `CBM.get_image_concept_names`
filters the segcrop-embedding dict by substring matching these basenames.

50 000 images in the full subset (1000 classes × 50). Seed 42 matches
upstream's `create_subsets.py`. Nested subsets: 5⊂10⊂25⊂50 per class.

Usage:
    python scripts/data_prep/imagenet_build_subsets.py
    python scripts/data_prep/imagenet_build_subsets.py --overwrite
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from cbm_in_the_wild.paths import imagenet_dir


DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))
OUT_DIR = DCBM_REPO / "data" / "Embeddings" / "subsets" / "ImageNet"
SEED = 42
SAMPLE_SIZES = [5, 10, 25, 50]
EXPECTED_CLASSES = 1000


def sample_subsets(train_root: Path) -> dict[int, list[str]]:
    random.seed(SEED)
    wnid_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    if len(wnid_dirs) != EXPECTED_CLASSES:
        raise ValueError(
            f"expected {EXPECTED_CLASSES} wnid dirs under {train_root}, "
            f"got {len(wnid_dirs)}"
        )

    buckets: dict[int, list[str]] = {s: [] for s in SAMPLE_SIZES}
    for wd in wnid_dirs:
        imgs = sorted(p.name for p in wd.glob("*.JPEG"))
        if len(imgs) < SAMPLE_SIZES[-1]:
            raise ValueError(
                f"{wd} has only {len(imgs)} images; need ≥{SAMPLE_SIZES[-1]}"
            )
        r50 = random.sample(imgs, 50)
        r25 = random.sample(r50, 25)
        r10 = random.sample(r25, 10)
        r5 = random.sample(r10, 5)
        buckets[50].extend(r50)
        buckets[25].extend(r25)
        buckets[10].extend(r10)
        buckets[5].extend(r5)
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    train_root = imagenet_dir() / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"{train_root} not found — run split_val first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    biggest = OUT_DIR / "ImageNet_rand_50.txt"
    if biggest.exists() and not args.overwrite:
        print(f"[skip] {biggest} exists (pass --overwrite to regenerate)")
        return 0

    print(f"train_root={train_root}")
    print(f"OUT_DIR={OUT_DIR}")
    subsets = sample_subsets(train_root)
    for size, names in subsets.items():
        out = OUT_DIR / f"ImageNet_rand_{size}.txt"
        with open(out, "w") as f:
            for n in names:
                f.write(n + "\n")
        print(f"[write] {out} ({len(names)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
