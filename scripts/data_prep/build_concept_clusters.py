"""Precompute concept-cluster membership for the DCBM-paper Figure-5 view.

For each frozen centroid (cluster), assign every SAM2 segcrop to its nearest
centroid (Euclidean — matches KMeans partitioning), then keep the top-N members
closest to the centroid. The closest member is the "median image" representative
(orange-border anchor in the paper). Output is a compact JSON the analysis
notebook renders without touching the multi-GB segcrop blob.

Faithful-to-bank notes:
  - KMeans clusters on raw CLIP segcrop embeddings; we reconstruct membership by
    L2-nearest-centroid (Voronoi partition). centroid_method='median'.
  - ImageNet class labels are wnids; resolved to readable names via the
    wnid->name table (the user-facing "name instead of nXXXXXXXX" request).

Usage:
  python scripts/data_prep/build_concept_clusters.py --dataset cifar10
  python scripts/data_prep/build_concept_clusters.py --dataset imagenet --top-n 12

Skip-if-exists unless --force. Writes:
  data/checkpoints/<dataset>_baseline/concept_clusters.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cbm_in_the_wild.concept_names import (  # noqa: E402
    _CROPS_DIRNAME,
    _SEGCROP_DATASET,
    _SEGCROP_FILENAME_TEMPLATE,
    _centroid_key,
)
from cbm_in_the_wild.dcbm_bridge import _resolve_dcbm_repo  # noqa: E402
from cbm_in_the_wild.imagenet_labels import wnid_to_name  # noqa: E402
from cbm_in_the_wild.paths import CHECKPOINT_DIR  # noqa: E402
from cbm_in_the_wild.persistence import load_cbm_artifacts  # noqa: E402


def build(dataset: str, *, top_n: int = 12, chunk_size: int = 20_000, force: bool = False) -> Path:
    out_path = CHECKPOINT_DIR / f"{dataset}_baseline" / "concept_clusters.json"
    artifacts = load_cbm_artifacts(CHECKPOINT_DIR / f"{dataset}_baseline")
    centroids: torch.Tensor = artifacts["centroids"].float()  # (K, m)
    K = centroids.shape[0]
    key = _centroid_key(centroids)

    if out_path.exists() and not force:
        cached = json.loads(out_path.read_text())
        if cached.get("centroid_key") == key and len(cached.get("clusters", [])) == K:
            print(f"[skip] {out_path} up to date (K={K})")
            return out_path

    repo = _resolve_dcbm_repo(None)
    seg_blob_path = (
        repo / "data" / "Segments"
        / _SEGCROP_FILENAME_TEMPLATE.format(ds=_SEGCROP_DATASET[dataset])
    )
    crops_root = repo / "data" / "Segments" / _CROPS_DIRNAME[dataset] / "crops"
    if not seg_blob_path.exists():
        raise FileNotFoundError(f"segcrop blob not found at {seg_blob_path}")

    print(f"[load] {seg_blob_path} ...")
    blob: dict = torch.load(seg_blob_path, map_location="cpu", weights_only=False)
    fnames = list(blob.keys())
    embs = torch.stack([blob[f][0].float() for f in fnames], dim=0)  # (N, m)
    labels = [blob[f][1] for f in fnames]
    N = embs.shape[0]
    print(f"[info] dataset={dataset} K={K} segcrops={N} top_n={top_n}")

    # Assign each crop to its nearest centroid (L2), tracking distance — chunked
    # over crops to bound memory on ImageNet (~330k crops).
    cent = centroids
    assign = torch.empty(N, dtype=torch.long)
    dist = torch.empty(N, dtype=torch.float32)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        d = torch.cdist(embs[s:e], cent)          # (B, K)
        dmin, amin = d.min(dim=1)
        assign[s:e] = amin
        dist[s:e] = dmin

    clusters = []
    for k in range(K):
        member_idx = (assign == k).nonzero(as_tuple=True)[0]
        if member_idx.numel() == 0:
            clusters.append({"centroid": k, "size": 0, "members": []})
            continue
        order = member_idx[torch.argsort(dist[member_idx])]  # closest first
        keep = order[:top_n]
        members = []
        for rank, i in enumerate(keep.tolist()):
            cls = labels[i]
            members.append({
                "path": str(crops_root / cls / fnames[i]),
                "class_wnid": cls,
                "class_name": wnid_to_name(cls) if dataset == "imagenet" else cls,
                "dist": round(float(dist[i]), 5),
                "is_representative": rank == 0,  # closest to centroid = median image
            })
        clusters.append({"centroid": k, "size": int(member_idx.numel()), "members": members})

    out_path.write_text(json.dumps({
        "dataset": dataset,
        "centroid_key": key,
        "K": K,
        "n_segcrops": N,
        "top_n": top_n,
        "metric": "l2_nearest_centroid",
        "clusters": clusters,
    }, indent=1))
    print(f"[write] {out_path}  ({K} clusters)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "imagenet"])
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(args.dataset, top_n=args.top_n, force=args.force)
