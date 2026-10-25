"""CLIP-ViT-B/16 image embeddings for ImageNet train/val/test.

Writes three files (one per split) into $DCBM_REPO/data/Embeddings/:
    images_ImageNet_{train,val,test}_CLIP-ViT-B16.torch

File schema (per split): { filename: [embedding_tensor<512>, wnid_str] }

Note the capital 'ImageNet' — upstream `utils/dcbm.py:203` uses that casing,
differently from the lowercase `cifar10`. The `CBM.construct_paths` call
matches file names case-sensitively, so we must align.

Label strings are wnids (e.g. `n01440764`), not human-readable class names.
DCBM's `load_labels_list` reads the human-name mapping from the class-labels
file separately, so the embedding dict only needs to record the wnid.

Skip-if-exists per split. ImageNet train is ~1.28M images; time on H100 is
~1.5 h at 140 imgs/s (empirically calibrated from CIFAR-10 extract speed,
adjusted for ImageNet's larger average file size). Route via
`gpu_h100_short` with `-t 2:00:00` (per `01_compute_budgets.md` §2.4).
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

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
from cbm_in_the_wild.clip_embed import compute_embeddings_batched, model_tag  # noqa: E402
from cbm_in_the_wild.paths import imagenet_dir  # noqa: E402


DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))
OUT_DIR = DCBM_REPO / "data" / "Embeddings"


def list_split_images(split: str) -> list[str]:
    split_dir = imagenet_dir() / split
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_dir} not found")
    return sorted(glob.glob(str(split_dir / "*" / "*.JPEG")))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default=None,
                   help="Single split to embed; default runs all three.")
    p.add_argument("--model", default="ViT-B/16",
                   help="CLIP model name (default: ViT-B/16). E.g. RN50.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Write a .ckpt.torch sidecar every N images for "
                        "resume across chained short SLURM segments.")
    args = p.parse_args()

    MODEL_NAME = args.model
    MODEL_TAG = model_tag(MODEL_NAME)
    splits = [args.split] if args.split else ["train", "val", "test"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"ImageNet root={imagenet_dir()}")
    print(f"OUT_DIR={OUT_DIR} model={MODEL_NAME} tag={MODEL_TAG}")

    print(f"loading CLIP {MODEL_NAME}…")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    for split in splits:
        out = OUT_DIR / f"images_ImageNet_{split}_{MODEL_TAG}.torch"
        if out.exists() and not args.overwrite:
            print(f"[skip] {out.name} exists (pass --overwrite to regenerate)")
            continue
        paths = list_split_images(split)
        print(f"[{split}] {len(paths)} images → {out.name}")
        t0 = time.time()
        compute_embeddings_batched(
            paths, model, preprocess, out, device, batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"[{split}] done in {time.time()-t0:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
