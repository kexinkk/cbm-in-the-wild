"""Concept-name lookup for DCBM bank centroids.

For each centroid $c_j$ in a Phase-1 bank, return a class-grounded label —
the class of the nearest training segcrop in CLIP-D space (cosine).

The DCBM paper §3.4 describes a richer
text-vocabulary-based naming, but upstream `DCBM/utils/dcbm.py` does not
ship that implementation; vocabulary choice would also be an open
methodology decision outside this study's scope. The labels here are what
the upstream artifacts can directly produce, deterministic and
reproducible.

The segcrop blob is dict-shaped (canonical artifact paths, README):

    $DCBM_REPO/data/Segments/segcrop_<Dataset>_SAM2_CLIP-ViT-B16.torch
        → {filename: [embedding (512,), class_label (str)]}

Per-dataset casing differs (upstream DCBM casing, see README): `cifar10` lowercase,
`ImageNet` PascalCase. Class labels are already human-readable for
CIFAR-10 (`'airplane'`, ...); ImageNet labels are wnids (`'n01440764'`),
which the caller resolves to readable names if needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from cbm_in_the_wild.dcbm_bridge import _resolve_dcbm_repo
from cbm_in_the_wild.paths import CHECKPOINT_DIR
from cbm_in_the_wild.persistence import load_cbm_artifacts


# Per-dataset segcrop filename casing — upstream DCBM casing, see README.
# CUB's segcrop token is CUB_200_2011 (not CUB) — the blob on disk is
# segcrop_CUB_200_2011_SAM2_CLIP-ViT-B16.torch (upstream dcbm.py:513).
_SEGCROP_DATASET = {"cifar10": "cifar10", "imagenet": "ImageNet", "cub": "CUB_200_2011",
                    "skincon": "skincon", "derm7pt": "derm7pt"}
# `{model}` is the slash-stripped CLIP tag — CLIP-ViT-B16 (512-d) or CLIP-RN50
# (1024-d). It MUST match the backbone of the centroids being named, or the
# cosine match is a dimension mismatch (canonical artifact paths, README).
_SEGCROP_FILENAME_TEMPLATE = "segcrop_{ds}_SAM2_{model}.torch"


def load_concept_names(
    dataset: str,
    *,
    model_tag: str = "CLIP-ViT-B16",
    baseline_name: str | None = None,
    dcbm_repo: Path | None = None,
    cache: bool = True,
    chunk_size: int = 50_000,
) -> list[str]:
    """Return one class label per centroid, indexed by centroid id.

    Args:
        dataset: ``'cifar10'`` or ``'imagenet'``.
        model_tag: slash-stripped CLIP tag selecting the segcrop blob —
            ``'CLIP-ViT-B16'`` (default, 512-d) or ``'CLIP-RN50'`` (1024-d).
            Must match the backbone of the bank in ``baseline_name``.
        baseline_name: checkpoint subdir under ``data/checkpoints/`` to load
            centroids + cache from (default ``'<dataset>_baseline'``, the ViT
            bank; pass e.g. ``'imagenet_rn50_baseline'`` for the RN50 bank).
        dcbm_repo: path to the upstream DCBM fork (default: $DCBM_REPO or ~/DCBM).
        cache: read/write a JSON cache at
            ``data/checkpoints/<baseline_name>/concept_names.json``.
            Cache key is the SHA-256 of the centroid tensor's bytes, so the
            cache invalidates if the bank changes.
        chunk_size: segcrop batch size for the cosine-argmax stream
            (memory-efficient on ImageNet's ~620k segcrops).

    Returns:
        ``names`` of length ``K`` where ``names[k]`` is the class of the
        training segcrop with maximum cosine similarity to centroid ``k``.
    """
    if dataset not in _SEGCROP_DATASET:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(_SEGCROP_DATASET)}")
    baseline_name = baseline_name or f"{dataset}_baseline"

    artifacts = load_cbm_artifacts(CHECKPOINT_DIR / baseline_name)
    centroids: torch.Tensor = artifacts["centroids"].float()  # (K, m)

    cache_path = CHECKPOINT_DIR / baseline_name / "concept_names.json"
    centroid_key = _centroid_key(centroids)
    if cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("centroid_key") == centroid_key and len(cached.get("names", [])) == centroids.shape[0]:
            return list(cached["names"])

    seg_blob_path = (
        _resolve_dcbm_repo(dcbm_repo) / "data" / "Segments"
        / _SEGCROP_FILENAME_TEMPLATE.format(ds=_SEGCROP_DATASET[dataset], model=model_tag)
    )
    if not seg_blob_path.exists():
        raise FileNotFoundError(f"segcrop blob not found at {seg_blob_path}")

    blob: dict = torch.load(seg_blob_path, map_location="cpu", weights_only=False)
    seg_embs = torch.stack([v[0].float() for v in blob.values()], dim=0)  # (N, m)
    seg_labels = [v[1] for v in blob.values()]                            # length N

    cn = torch.nn.functional.normalize(centroids, dim=-1)                 # (K, m)

    K = centroids.shape[0]
    best_sim = torch.full((K,), float("-inf"))
    best_seg_idx = torch.zeros(K, dtype=torch.long)

    for start in range(0, seg_embs.shape[0], chunk_size):
        end = min(start + chunk_size, seg_embs.shape[0])
        chunk = torch.nn.functional.normalize(seg_embs[start:end], dim=-1)  # (B, m)
        sim = cn @ chunk.t()                                                # (K, B)
        chunk_max, chunk_arg = sim.max(dim=-1)                              # (K,), (K,)
        update = chunk_max > best_sim
        best_seg_idx[update] = chunk_arg[update] + start
        best_sim[update] = chunk_max[update]

    names = [seg_labels[i] for i in best_seg_idx.tolist()]

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"centroid_key": centroid_key, "names": names, "n": len(names)},
            indent=2,
        ))

    return names


def _centroid_key(centroids: torch.Tensor) -> str:
    """SHA-256 of the centroid tensor's bytes — cache invalidator."""
    import hashlib
    h = hashlib.sha256(centroids.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# Per-dataset segcrop dirname (the SAM2-output dir, not the blob file).
# CIFAR uses lowercase; ImageNet uses lowercase too here (see README
# canonical-paths note: the `<dataset>_<seg>/crops/` dir is the exception —
# lowercase across the board).
_CROPS_DIRNAME = {"cifar10": "cifar10_SAM2", "imagenet": "imagenet_SAM2", "cub": "CUB_200_2011_SAM2",
                  "skincon": "skincon_SAM2", "derm7pt": "derm7pt_SAM2"}


def load_centroid_representative_crops(
    dataset: str,
    *,
    model_tag: str = "CLIP-ViT-B16",
    baseline_name: str | None = None,
    dcbm_repo: Path | None = None,
    cache: bool = True,
    chunk_size: int = 50_000,
) -> dict[int, Path]:
    """Return ``{centroid_k: Path}`` mapping each centroid to its nearest segcrop
    image file (cosine-argmax in CLIP-D space, mirrors ``load_concept_names``
    but keeps the *filename* instead of dropping it for the class label).

    ``model_tag`` / ``baseline_name`` select the backbone+bank exactly as in
    ``load_concept_names`` (default = the ViT-B16 ``<dataset>_baseline``).

    Used by ``scripts/data_prep/render_concept_crops.py`` to render the
    per-dataset companion sheet, and by visualization helpers that want to
    show *what each centroid actually looks like* alongside heatmaps —
    distinguishes e.g. cifar10 c95 vs c13 (both labelled "airplane" but
    representing different airplane sub-parts).
    """
    if dataset not in _SEGCROP_DATASET:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(_SEGCROP_DATASET)}")
    baseline_name = baseline_name or f"{dataset}_baseline"

    artifacts = load_cbm_artifacts(CHECKPOINT_DIR / baseline_name)
    centroids: torch.Tensor = artifacts["centroids"].float()

    cache_path = CHECKPOINT_DIR / baseline_name / "centroid_crops.json"
    centroid_key = _centroid_key(centroids)
    if cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("centroid_key") == centroid_key and len(cached.get("crops", [])) == centroids.shape[0]:
            return {int(k): Path(v) for k, v in cached["crops"].items()}

    repo = _resolve_dcbm_repo(dcbm_repo)
    seg_blob_path = (
        repo / "data" / "Segments"
        / _SEGCROP_FILENAME_TEMPLATE.format(ds=_SEGCROP_DATASET[dataset], model=model_tag)
    )
    if not seg_blob_path.exists():
        raise FileNotFoundError(f"segcrop blob not found at {seg_blob_path}")
    crops_root = repo / "data" / "Segments" / _CROPS_DIRNAME[dataset] / "crops"

    blob: dict = torch.load(seg_blob_path, map_location="cpu", weights_only=False)
    seg_filenames = list(blob.keys())
    seg_embs = torch.stack([blob[f][0].float() for f in seg_filenames], dim=0)
    seg_labels = [blob[f][1] for f in seg_filenames]

    cn = torch.nn.functional.normalize(centroids, dim=-1)
    K = centroids.shape[0]
    best_sim = torch.full((K,), float("-inf"))
    best_idx = torch.zeros(K, dtype=torch.long)

    for start in range(0, seg_embs.shape[0], chunk_size):
        end = min(start + chunk_size, seg_embs.shape[0])
        chunk = torch.nn.functional.normalize(seg_embs[start:end], dim=-1)
        sim = cn @ chunk.t()
        chunk_max, chunk_arg = sim.max(dim=-1)
        update = chunk_max > best_sim
        best_idx[update] = chunk_arg[update] + start
        best_sim[update] = chunk_max[update]

    crops: dict[int, Path] = {}
    for k in range(K):
        i = int(best_idx[k])
        fn = seg_filenames[i]
        cls = seg_labels[i]
        crops[k] = crops_root / cls / fn

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "centroid_key": centroid_key,
            "crops": {str(k): str(p) for k, p in crops.items()},
            "n": len(crops),
        }, indent=2))

    return crops
