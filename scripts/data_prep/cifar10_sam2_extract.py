"""Step 4: SAM2 segmentation on the 500-image CIFAR-10 concept subset.

Reads $DCBM_REPO/data/Embeddings/subsets/cifar10/cifar10_rand_50.txt
(produced by cifar10_build_subsets.py — must run first), resolves each
filename to its full path under $CIFAR10_ROOT/train/<class>/<fname>,
and runs SAM2 per-image.

CIFAR-10 is 32×32 — SAM2 is trained at ≥1024 px and `points_per_side=64`
would otherwise place prompts at <0.5 pixel spacing on the source. We
monkey-patch `utilities.SAM2.resize_image` to upsample 32→1024 with
bicubic interpolation *in memory* (no disk persistence). Without it, the
upstream resize is a no-op below 1024 px.

Crops land at:
    $DCBM_REPO/data/Segments/cifar10_SAM2/crops/<class>/<imgstem>_OUT_CROP_<i>_X_.jpg
    $DCBM_REPO/data/Segments/cifar10_SAM2/masks/<class>/<imgstem>_OUT_MASK_<i>_X_.jpg
Stats (per-image bounding boxes, sizes) at:
    $DCBM_REPO/data/Segments/stats/cifar10_SAM2_STATS.torch

Sharding: set SHARD_ID / SHARD_COUNT env vars to process a slice of the subset.
Each shard writes its own partial stats file which we merge on the final
shard; crops/masks are written to a shared dir (restartable via skip-
if-exists).

Usage:
    python scripts/data_prep/cifar10_sam2_extract.py
    SHARD_ID=0 SHARD_COUNT=4 python scripts/data_prep/cifar10_sam2_extract.py   # one of 4 shards
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import tqdm

DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))
CIFAR10_ROOT = Path(
    os.environ.get(
        "CIFAR10_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "cifar10",
    )
)

SEGMENTS_ROOT = DCBM_REPO / "data" / "Segments"
SUBSET_FILE = DCBM_REPO / "data" / "Embeddings" / "subsets" / "cifar10" / "cifar10_rand_50.txt"
OUT_PATH = SEGMENTS_ROOT / "cifar10_SAM2"
STATS_DIR = SEGMENTS_ROOT / "stats"
STATS_FILE = STATS_DIR / "cifar10_SAM2_STATS.torch"

# sam2 package ships configs at configs/sam2.1/… inside its installed package —
# hydra resolves this relative to the sam2 module root at runtime.
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = (
    SEGMENTS_ROOT / "scripts" / "utilities" / "segment_anything_2"
    / "checkpoints" / "sam2.1_hiera_large.pt"
)

UPSAMPLE_TARGET = 1024  # see decision doc


def _install_upstream_import_path() -> None:
    """Put DCBM/data/Segments/scripts on sys.path so utilities.* imports work.

    `utilities.SAM2` uses `from .segment import …`, so `utilities` must be
    loadable as a (namespace) package — the scripts/ dir is its parent.
    """
    scripts_dir = str(SEGMENTS_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _patch_resize_for_cifar() -> None:
    """Replace `utilities.SAM2.resize_image` so 32×32 upsamples instead of no-op."""
    from utilities import SAM2 as sam2_mod  # noqa: E402

    orig = sam2_mod.resize_image

    def cifar_aware_resize(image, target_size=UPSAMPLE_TARGET):
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side < target_size:
            # upsample (CIFAR-10 path)
            scale = target_size / long_side
            return cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        return orig(image, target_size)  # delegate downsample path unchanged

    sam2_mod.resize_image = cifar_aware_resize


def build_subset_paths(subset_file: Path, cifar_root: Path) -> list[str]:
    """Resolve subset basenames to absolute paths via the ImageFolder tree."""
    if not subset_file.exists():
        raise FileNotFoundError(
            f"subset file {subset_file} missing — run cifar10_build_subsets.py first"
        )
    with open(subset_file) as f:
        basenames = [line.strip() for line in f if line.strip()]

    train_root = cifar_root / "train"
    name2path = {p.name: str(p) for p in train_root.rglob("*.png")}
    missing = [n for n in basenames if n not in name2path]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} subset filenames not found under {train_root}: "
            f"{missing[:5]}…"
        )
    return [name2path[n] for n in basenames]


def shard(items: list, shard_id: int, shard_count: int) -> list:
    return items[shard_id::shard_count]


def already_processed(img_path: str, out_path: Path) -> bool:
    """True if any crop for this image already exists on disk."""
    ifolder = Path(img_path).parent.name
    istem = Path(img_path).stem
    # create_crops_masks writes `{istem}_OUT_CROP_<i>_X_.jpg`
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
    _patch_resize_for_cifar()

    from utilities.SAM2 import generate_mask  # noqa: E402 — after sys.path + patch
    from sam2.build_sam import build_sam2  # noqa: E402

    if not SAM2_CKPT.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {SAM2_CKPT}. "
            "Run scripts/utilities/segment_anything_2/checkpoints/download_ckpts.sh."
        )

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"shard={args.shard_id}/{args.shard_count}")
    print(f"SEGMENTS_ROOT={SEGMENTS_ROOT}")
    print(f"CIFAR10_ROOT={CIFAR10_ROOT}")

    all_paths = build_subset_paths(SUBSET_FILE, CIFAR10_ROOT)
    my_paths = shard(all_paths, args.shard_id, args.shard_count)
    print(f"subset: {len(all_paths)} total, {len(my_paths)} this shard")

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    # pre-create per-class crop + mask dirs so generate_mask's makedirs calls
    # don't race across shards
    classes = sorted({Path(p).parent.name for p in all_paths})
    for cls in classes:
        (OUT_PATH / "crops" / cls).mkdir(parents=True, exist_ok=True)
        (OUT_PATH / "masks" / cls).mkdir(parents=True, exist_ok=True)

    # dataset label passed to create_statistics — "cifar10" uses the simple
    # bbox/sizes path (no CUB-style overlap-with-gt branch)
    dataset_label = "cifar10"

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
    print(f"shard done in {elapsed:.1f}s "
          f"({elapsed / max(1, len(my_paths)):.2f}s/img), "
          f"no-mask: {total_no_masks}")

    # Merge into the shared stats file. We key by ipath (basename stem),
    # and add the totals under fixed keys. Non-atomic; if shards collide
    # last-write wins for the totals — acceptable for our 500-image scale.
    shard_stats_file = (
        STATS_FILE if args.shard_count == 1
        else STATS_DIR / f"cifar10_SAM2_STATS_shard{args.shard_id}of{args.shard_count}.torch"
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
