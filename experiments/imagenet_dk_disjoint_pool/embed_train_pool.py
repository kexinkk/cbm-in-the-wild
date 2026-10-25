#!/usr/bin/env python
"""CLIP ViT-B/16 embeddings of the corrupted disjoint train pool.

For each of the 10 corruptions x 5 severities, corrupts the 50,000 ImageNet train
images in train_pool_ids.txt (50 per class, disjoint from the 50k evaluation images
and the 128,118-image refit pool) and writes
    $DK_POOL_CELL_ROOT/<corruption>_sev<severity>.pt = {"ids", "clip_emb" [N,512] f32, "hashseed"}
which compute_dk.py reads. Skips cells that already exist.

Launch with PYTHONHASHSEED=0: the per-image corruption seed is a Python string hash
(see cbm_in_the_wild.train_pool), and 0 is the seed of the released numbers.

Usage:
    PYTHONHASHSEED=0 python experiments/imagenet_dk_disjoint_pool/embed_train_pool.py [--task i]
    PYTHONHASHSEED=0 python experiments/imagenet_dk_disjoint_pool/embed_train_pool.py --verify REF_DIR
(--task i runs only corruption i, all severities, for SLURM arrays; --verify compares the
first --n images of every cell against an existing cell directory and writes nothing.)
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from cbm_in_the_wild.corruption_eval import (  # noqa: E402
    _clip_forward_factory, _load_clip, resolve_corruptions,
)
from cbm_in_the_wild.paths import DATA_DIR  # noqa: E402
from cbm_in_the_wild.train_pool import corrupted_train_image  # noqa: E402

CELL_ROOT = Path(os.environ.get("DK_POOL_CELL_ROOT", DATA_DIR / "dk_disjoint_pool" / "train_cells"))
SEVERITIES = [1, 2, 3, 4, 5]
BATCH = 512


def _work(args):
    torch.set_num_threads(1)
    ids, corruption, severity = args
    return np.stack([corrupted_train_image(i, i.split("_")[0], corruption, severity) for i in ids])


def embed(ids, corruption, severity, fwd, pool) -> torch.Tensor:
    chunks = [(ids[k:k + 32], corruption, severity) for k in range(0, len(ids), 32)]
    out, buf = [], []
    for arr in pool.map(_work, chunks):
        buf.append(arr)
        if sum(len(a) for a in buf) >= BATCH:
            out.append(fwd(np.concatenate(buf))); buf = []
    if buf:
        out.append(fwd(np.concatenate(buf)))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--verify", type=Path, default=None)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("launch with PYTHONHASHSEED=0 (per-image corruption seeds are string hashes)")

    ids = (HERE / "train_pool_ids.txt").read_text().split()
    corruptions = resolve_corruptions("combined")
    if args.task is not None:
        corruptions = [corruptions[args.task]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, pre = _load_clip(device, clip_arch="ViT-B/16")
    fwd = _clip_forward_factory(model, pre, device)

    with ProcessPoolExecutor(args.workers) as pool:
        for c in corruptions:
            for s in SEVERITIES:
                name = f"{c}_sev{s}.pt"
                if args.verify is not None:
                    ref = torch.load(args.verify / name, map_location="cpu", weights_only=False)
                    assert list(ref["ids"]) == ids, f"id order differs in {name}"
                    z = embed(ids[:args.n], c, s, fwd, pool)
                    cos = torch.nn.functional.cosine_similarity(z, ref["clip_emb"][:args.n].float())
                    print(f"{name}: cos mean {cos.mean():.6f} min {cos.min():.6f}", flush=True)
                    continue
                out = CELL_ROOT / name
                if out.exists():
                    print(f"{name} exists, skipping"); continue
                out.parent.mkdir(parents=True, exist_ok=True)
                z = embed(ids, c, s, fwd, pool)
                torch.save({"ids": ids, "clip_emb": z, "hashseed": "0"}, str(out) + ".tmp")
                os.replace(str(out) + ".tmp", out)
                print(f"wrote {out} {tuple(z.shape)}", flush=True)


if __name__ == "__main__":
    main()
