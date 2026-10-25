"""Centralized path resolution for data, checkpoints, and experiment outputs.

All paths can be overridden via environment variables, which is the
preferred mechanism on a SLURM cluster where scratch paths differ per user.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("DCBM_DATA_DIR", REPO_ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get("DCBM_CKPT_DIR", DATA_DIR / "checkpoints"))
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def cifar10_dir() -> Path:
    return DATA_DIR / "cifar10"


def cifar10_c_dir() -> Path:
    return DATA_DIR / "raw" / "CIFAR-10-C"


def cifar10_p_dir() -> Path:
    return DATA_DIR / "raw" / "CIFAR-10-P"


def imagenet_dir() -> Path:
    # DCBM_IMAGENET_DIR redirects only the clean ImageNet tree (e.g. a
    # $TMPDIR-staged test-only copy for corruption-eval sensor cells) without
    # moving DATA_DIR — so the deployed bank under DATA_DIR/checkpoints stays
    # on Lustre. Mirrors the DCBM_IMGNET_C_DIR knob below.
    override = os.environ.get("DCBM_IMAGENET_DIR")
    if override:
        return Path(override)
    return DATA_DIR / "imagenet"


def imagenet_image_path(split: str, wnid: str, image_id: str) -> Path:
    """Path of one ImageNet image. `test` = ILSVRC val tree. `train`/`val` were
    carved from train_full: their symlink trees were replaced by id manifests
    (`imagenet/splits/{train,val}.txt`, 2026-10-03), so fall back to train_full
    when the split dir is absent. Staged tars that still carry a real
    `train/` layout (imagenet_subset.tar) resolve directly."""
    root = imagenet_dir()
    p = root / split / wnid / image_id
    if split in ("train", "val") and not (root / split).exists():
        return root / "train_full" / wnid / image_id
    return p


def cub_dir() -> Path:
    # DCBM_CUB_DIR redirects the clean CUB-200-2011 tree to a $TMPDIR-staged
    # copy (tar-extracted on local SSD) so no stage iterates the loose ~12k-JPEG
    # tree on Lustre. Mirrors the DCBM_IMAGENET_DIR knob; DATA_DIR is unaffected,
    # so banks/embeddings under DATA_DIR stay on the workspace.
    override = os.environ.get("DCBM_CUB_DIR")
    if override:
        return Path(override)
    return DATA_DIR / "CUB_200_2011"


def imagenet_c_dir() -> Path:
    override = os.environ.get("DCBM_IMGNET_C_DIR")
    if override:
        return Path(override)
    return DATA_DIR / "raw" / "ImageNet-C"


def imagenet_p_dir() -> Path:
    return DATA_DIR / "raw" / "ImageNet-P"
