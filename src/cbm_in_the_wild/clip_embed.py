"""Batched CLIP image embedding — correctness-fixed replacement for upstream.

The upstream `compute_embeddings_batched` in
`DCBM/data/Datasets/scripts/utilities/clip.py` has two bugs that silently
produce corrupted embeddings:
  1. References an undefined name `image_keys` for batch key slicing.
  2. Inside the batch inner loop it indexes `image_paths[i]` (not `[x]`),
     so every path in the batch ends up being the first path.

Upstream's non-batched `compute_embeddings` is correct but one-image-at-a-time.
We reimplement the batched version cleanly; it's the same ~30 lines.

Output schema matches upstream exactly so the `CBM` class consumes our files
without change:
    { filename: [embedding_tensor_1d, class_name_str] }
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import tqdm
from PIL import Image, ImageFile

# Some CIFAR-10 PNGs in the wild are subtly truncated by the pickle→PNG
# conversion step. Upstream flips this same flag in the same place.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def model_tag(model_name: str) -> str:
    """Derive the filename-safe model tag from a CLIP model name.

    ``"ViT-B/16"`` → ``"CLIP-ViT-B16"``; ``"RN50"`` → ``"CLIP-RN50"``.
    """
    return "CLIP-" + model_name.replace("/", "")


def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("P", "RGBA"):
        return img.convert("RGBA")
    return img.convert("RGB")


def compute_embeddings_batched(
    image_paths: Iterable[str],
    model,
    preprocess,
    output_file: str | Path,
    device: str | torch.device,
    *,
    batch_size: int = 64,
    key_fn=lambda p: p.split("/")[-1],
    class_fn=lambda p: p.split("/")[-2],
    checkpoint_every: int = 0,
) -> dict:
    """Embed `image_paths` with CLIP `model` in batches and save to `output_file`.

    `key_fn` maps each path to the dict key (default: basename); `class_fn`
    maps each path to the class label string (default: parent dir name).
    Override these for non-ImageFolder layouts (e.g., SAM2 crops nested
    `crops/<class>/<img_OUT_CROP_i_X_>.jpg`).

    When ``checkpoint_every > 0``, a ``.ckpt.torch`` sidecar is written every
    N images and loaded on restart so chained short SLURM jobs can resume
    where the previous one left off.  The checkpoint is deleted once the
    final file is written.

    Returns the embeddings dict (also written to `output_file` via torch.save).
    """
    paths = list(image_paths)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ckpt_file = output_file.with_suffix(".ckpt.torch")
    if checkpoint_every > 0 and ckpt_file.exists():
        print(f"[resume] loading checkpoint {ckpt_file.name} …", flush=True)
        embeddings: dict = torch.load(ckpt_file, weights_only=False)
        print(f"[resume] {len(embeddings)} images already embedded", flush=True)
    else:
        embeddings = {}

    done_keys = set(embeddings.keys())
    imgs_since_ckpt = 0

    with torch.no_grad():
        for i in tqdm.tqdm(
            range(0, len(paths), batch_size), desc=f"embedding → {output_file.name}"
        ):
            batch_paths = paths[i : i + batch_size]
            # Skip fully-completed batches on resume.
            if done_keys and all(key_fn(p) in done_keys for p in batch_paths):
                continue
            batch_tensors = [preprocess(_load_image(p)).unsqueeze(0) for p in batch_paths]
            batch = torch.cat(batch_tensors).to(device)
            batch_embs = model.encode_image(batch).cpu()
            for j, p in enumerate(batch_paths):
                embeddings[key_fn(p)] = [batch_embs[j], class_fn(p)]
            imgs_since_ckpt += len(batch_paths)

            if checkpoint_every > 0 and imgs_since_ckpt >= checkpoint_every:
                torch.save(embeddings, ckpt_file)
                imgs_since_ckpt = 0

    torch.save(embeddings, output_file)
    if ckpt_file.exists():
        ckpt_file.unlink()
    return embeddings
