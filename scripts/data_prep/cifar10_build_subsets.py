"""Step 6 (order-wise; runs before SAM2): build 50/25/10/5-per-class concept subsets.

The `CBM` class loads these subsets via `get_image_concept_names`, which reads
    $DCBM_REPO/data/Embeddings/subsets/cifar10/cifar10_rand_50.txt
and filters the segment-crop embeddings down to 50 images per class (500 total
for CIFAR-10).

SAM2 extraction also reads this file — upstream `utilities.data_loading.load_data`
calls `extract_subset` which requires it — so this step must run BEFORE SAM2,
not after. The plan doc calls it "step 6" by numbering but in practice it's a
prerequisite for step 4.

Seed 42 matches upstream's `create_subsets.py`. File schema: one filename per
line, no path, no extension stripping. Example: `12345.png`.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))
CIFAR10_ROOT = Path(
    os.environ.get(
        "CIFAR10_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "cifar10",
    )
)
OUT_DIR = DCBM_REPO / "data" / "Embeddings" / "subsets" / "cifar10"
SEED = 42
SAMPLE_SIZES = [5, 10, 25, 50]
EXPECTED_CLASSES = 10


def sample_subsets(train_root: Path) -> dict[int, list[str]]:
    """Sample nested 5⊂10⊂25⊂50 subsets per class, return dict size→basenames."""
    random.seed(SEED)
    class_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    if len(class_dirs) != EXPECTED_CLASSES:
        raise ValueError(
            f"expected {EXPECTED_CLASSES} classes under {train_root}, "
            f"found {len(class_dirs)}: {[p.name for p in class_dirs]}"
        )

    rand50, rand25, rand10, rand5 = [], [], [], []
    for cls_dir in class_dirs:
        imgs = sorted(glob.glob(str(cls_dir / "*.png")))
        if len(imgs) < 50:
            raise ValueError(f"{cls_dir} has only {len(imgs)} images; need ≥50")
        r50 = random.sample(imgs, 50)
        r25 = random.sample(r50, 25)
        r10 = random.sample(r25, 10)
        r5 = random.sample(r10, 5)
        rand50.extend(r50)
        rand25.extend(r25)
        rand10.extend(r10)
        rand5.extend(r5)

    return {5: rand5, 10: rand10, 25: rand25, 50: rand50}


def write_subset(paths: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths:
            f.write(p.split("/")[-1] + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    train_root = CIFAR10_ROOT / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"{train_root} not found")

    print(f"CIFAR10_ROOT={CIFAR10_ROOT}")
    print(f"OUT_DIR={OUT_DIR}")

    # If the biggest subset already exists, upstream would interpret that as
    # "re-sample from existing" — we don't want to run twice. Bail instead.
    biggest = OUT_DIR / "cifar10_rand_50.txt"
    if biggest.exists() and not args.overwrite:
        print(f"[skip] {biggest} exists (pass --overwrite to regenerate)")
        return 0

    subsets = sample_subsets(train_root)
    for size, paths in subsets.items():
        out_path = OUT_DIR / f"cifar10_rand_{size}.txt"
        write_subset(paths, out_path)
        print(f"[write] {out_path} ({len(paths)} entries)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
