"""CLIP-ViT-B/16 embed SAM2 crops for ImageNet.

Globs crops produced by imagenet_sam2_extract.py:
    $DCBM_REPO/data/Segments/imagenet_SAM2/crops/<wnid>/<stem>_OUT_CROP_<i>_X_.jpg
Writes:
    $DCBM_REPO/data/Segments/segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch

Dict schema: { crop_filename: [embedding_tensor<512>, wnid_str] }.
Mirrors cifar10_segcrop_embed.py exactly — the only difference is paths
and the wnid-vs-class-name class label (same as imagenet_clip_extract).
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


DCBM_REPO = Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))
CROPS_ROOT = DCBM_REPO / "data" / "Segments" / "imagenet_SAM2" / "crops"


def list_crops() -> list[str]:
    if not CROPS_ROOT.exists():
        raise FileNotFoundError(
            f"crops dir {CROPS_ROOT} missing — run imagenet_sam2_extract.py first"
        )
    return sorted(glob.glob(str(CROPS_ROOT / "*" / "*.jpg")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B/16",
                    help="CLIP model name (default: ViT-B/16). E.g. RN50.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    MODEL_NAME = args.model
    MODEL_TAG = model_tag(MODEL_NAME)
    OUT_FILE = DCBM_REPO / "data" / "Segments" / f"segcrop_ImageNet_SAM2_{MODEL_TAG}.torch"

    if OUT_FILE.exists() and not args.overwrite:
        print(f"[skip] {OUT_FILE} exists (pass --overwrite to regenerate)")
        return 0

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")

    crops = list_crops()
    print(f"found {len(crops)} crops in {CROPS_ROOT}")
    if not crops:
        raise RuntimeError("no crops found — SAM2 extraction produced 0 outputs")

    print(f"loading CLIP {MODEL_NAME}…")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    t0 = time.time()
    compute_embeddings_batched(
        crops, model, preprocess, OUT_FILE, device, batch_size=args.batch_size
    )
    print(f"done in {time.time()-t0:.1f}s → {OUT_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
