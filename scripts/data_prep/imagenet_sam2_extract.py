"""SAM2 segmentation on the 50 000-image ImageNet concept subset.

Mirrors cifar10_sam2_extract.py with two deltas:
  1. No upsample monkey-patch — ImageNet JPEGs are ≥224 px (typically
     300–500 px on the long side) so SAM2's native `resize_image` path
     (downsample-to-1024-if-larger, else identity) is correct.
  2. Shard-aware from the start. ~50 000 images × ~4 s/image = ~55 GPU-h
     single-GPU. Per `01_compute_budgets.md` §2.5, using `--gres=gpu:4`
     drops this to ~14 h wall-clock on `gpu_a100_il` (48 h cap).

Reads:
    $DCBM_REPO/data/Embeddings/subsets/ImageNet/ImageNet_rand_50.txt
Writes:
    $DCBM_REPO/data/Segments/imagenet_SAM2/crops/<wnid>/<stem>_OUT_CROP_<i>_X_.jpg
    $DCBM_REPO/data/Segments/imagenet_SAM2/masks/<wnid>/<stem>_OUT_MASK_<i>_X_.jpg
    $DCBM_REPO/data/Segments/stats/imagenet_SAM2_STATS[_shardNofM].torch

Usage:
    # single GPU smoke (not expected to finish — use shards)
    python scripts/data_prep/imagenet_sam2_extract.py --shard-id 0 --shard-count 1
    # 4-way in-node parallelism under one sbatch
    SHARD_COUNT=4 SHARD_ID=0 python scripts/data_prep/imagenet_sam2_extract.py &
    SHARD_COUNT=4 SHARD_ID=1 python scripts/data_prep/imagenet_sam2_extract.py &
    ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import tqdm

from cbm_in_the_wild.paths import imagenet_dir


DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))
SEGMENTS_ROOT = DCBM_REPO / "data" / "Segments"
SUBSET_FILE = DCBM_REPO / "data" / "Embeddings" / "subsets" / "ImageNet" / "ImageNet_rand_50.txt"
OUT_PATH = SEGMENTS_ROOT / "imagenet_SAM2"
STATS_DIR = SEGMENTS_ROOT / "stats"
STATS_FILE = STATS_DIR / "imagenet_SAM2_STATS.torch"

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = (
    SEGMENTS_ROOT / "scripts" / "utilities" / "segment_anything_2"
    / "checkpoints" / "sam2.1_hiera_large.pt"
)


def _install_upstream_import_path() -> None:
    scripts_dir = str(SEGMENTS_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def build_subset_paths(subset_file: Path, train_root: Path) -> list[str]:
    if not subset_file.exists():
        raise FileNotFoundError(
            f"subset file {subset_file} missing — "
            "run imagenet_build_subsets.py first"
        )
    with open(subset_file) as f:
        basenames = [line.strip() for line in f if line.strip()]

    # ImageNet file stems embed the wnid (e.g. n01440764_10026.JPEG), so we
    # can resolve without a full rglob — split on "_" to get the wnid dir.
    # Fall back to a full scan only if that shortcut fails.
    paths: list[str] = []
    missing: list[str] = []
    for name in basenames:
        wnid = name.split("_")[0]
        p = train_root / wnid / name
        if p.exists():
            paths.append(str(p))
        else:
            missing.append(name)

    if missing:
        print(f"[warn] {len(missing)} basenames not found via wnid shortcut; "
              "falling back to rglob")
        name2path = {p.name: str(p) for p in train_root.rglob("*.JPEG")}
        for name in missing:
            if name in name2path:
                paths.append(name2path[name])
            else:
                raise FileNotFoundError(
                    f"subset basename {name} not found anywhere under {train_root}"
                )
    return paths


def shard(items: list, shard_id: int, shard_count: int) -> list:
    return items[shard_id::shard_count]


def already_processed(img_path: str, out_path: Path) -> bool:
    ifolder = Path(img_path).parent.name
    istem = Path(img_path).stem
    crop_dir = out_path / "crops" / ifolder
    if not crop_dir.exists():
        return False
    for f in crop_dir.iterdir():
        if f.name.startswith(f"{istem}_OUT_CROP_"):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-id", type=int, default=int(os.environ.get("SHARD_ID", 0)))
    ap.add_argument("--shard-count", type=int, default=int(os.environ.get("SHARD_COUNT", 1)))
    args = ap.parse_args()

    _install_upstream_import_path()

    from utilities.SAM2 import generate_mask  # noqa: E402 — after sys.path tweak
    from sam2.build_sam import build_sam2  # noqa: E402

    if not SAM2_CKPT.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {SAM2_CKPT}")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"shard={args.shard_id}/{args.shard_count}")
    print(f"SEGMENTS_ROOT={SEGMENTS_ROOT}")

    train_root = imagenet_dir() / "train"
    all_paths = build_subset_paths(SUBSET_FILE, train_root)
    my_paths = shard(all_paths, args.shard_id, args.shard_count)
    print(f"subset: {len(all_paths)} total, {len(my_paths)} this shard")

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    wnids = sorted({Path(p).parent.name for p in all_paths})
    for wnid in wnids:
        (OUT_PATH / "crops" / wnid).mkdir(parents=True, exist_ok=True)
        (OUT_PATH / "masks" / wnid).mkdir(parents=True, exist_ok=True)

    dataset_label = "imagenet"

    print("loading SAM2…")
    sam2 = build_sam2(SAM2_CONFIG, str(SAM2_CKPT), device=device, apply_postprocessing=False)
    sam2 = sam2.to(device)

    stats: dict = {}
    total_no_masks = 0
    total_img_wo_masks: list = []
    t0 = time.time()

    for img_path in tqdm.tqdm(my_paths, desc="SAM2"):
        if already_processed(img_path, OUT_PATH):
            continue
        out = generate_mask(
            sam2, img_path, str(OUT_PATH),
            create_stats=True, create_segments=True,
            dataset_name=dataset_label,
        )
        if out is None:
            continue
        s, ipath, no_masks_count, img_wo_masks = out
        stats[ipath] = s
        total_no_masks += no_masks_count
        total_img_wo_masks += img_wo_masks

    elapsed = time.time() - t0
    per = elapsed / max(1, len(my_paths))
    print(f"shard done in {elapsed:.1f}s ({per:.2f}s/img), "
          f"no-mask: {total_no_masks}")

    shard_stats_file = (
        STATS_FILE if args.shard_count == 1
        else STATS_DIR / f"imagenet_SAM2_STATS_shard{args.shard_id}of{args.shard_count}.torch"
    )
    stats["total_no_masks_count"] = total_no_masks
    stats["total_img_wo_masks"] = total_img_wo_masks
    if shard_stats_file.exists():
        prior = torch.load(shard_stats_file, weights_only=False)
        prior.update(stats)
        stats = prior
    torch.save(stats, shard_stats_file)
    print(f"stats → {shard_stats_file} ({len(stats)} entries)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
