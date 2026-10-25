"""Step 3: CLIP-ViT-B/16 image embeddings for CIFAR-10 train/val/test.

Writes three files (one per split) into `$DCBM_REPO/data/Embeddings/`:
    images_cifar10_{train,val,test}_CLIP-ViT-B16.torch

The `CBM` class loads these via `construct_paths` when `embeddings_path` is
set to `$DCBM_REPO/data/Embeddings/`. File schema (per split):
    { filename: [embedding_tensor<512>, class_name_str] }

Usage (from repo root, env already active):
    python scripts/data_prep/cifar10_clip_extract.py              # all splits
    python scripts/data_prep/cifar10_clip_extract.py --split train

Skip-if-exists: if the output file for a split already exists, that split
is skipped. Pass --overwrite to force re-extraction.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import clip
import torch

# Import our corrected batched embedder
REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
from cbm_in_the_wild.clip_embed import compute_embeddings_batched, model_tag  # noqa: E402


DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))
CIFAR10_ROOT = Path(
    os.environ.get(
        "CIFAR10_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "cifar10",
    )
)
OUT_DIR = DCBM_REPO / "data" / "Embeddings"


def list_split_images(split: str) -> list[str]:
    split_dir = CIFAR10_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_dir} not found")
    return sorted(glob.glob(str(split_dir / "*" / "*.png")))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default=None,
                   help="Single split to embed; default runs all three.")
    p.add_argument("--model", default="ViT-B/16",
                   help="CLIP model name (default: ViT-B/16). E.g. RN50.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    MODEL_NAME = args.model
    MODEL_TAG = model_tag(MODEL_NAME)
    splits = [args.split] if args.split else ["train", "val", "test"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"CIFAR10_ROOT={CIFAR10_ROOT}")
    print(f"OUT_DIR={OUT_DIR} model={MODEL_NAME} tag={MODEL_TAG}")

    print(f"loading CLIP {MODEL_NAME}…")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    for split in splits:
        out = OUT_DIR / f"images_cifar10_{split}_{MODEL_TAG}.torch"
        if out.exists() and not args.overwrite:
            print(f"[skip] {out.name} exists (pass --overwrite to regenerate)")
            continue
        paths = list_split_images(split)
        print(f"[{split}] {len(paths)} images → {out.name}")
        t0 = time.time()
        compute_embeddings_batched(
            paths, model, preprocess, out, device, batch_size=args.batch_size
        )
        print(f"[{split}] done in {time.time()-t0:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
