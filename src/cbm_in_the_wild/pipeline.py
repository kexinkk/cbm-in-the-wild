"""Clean / corrupted propagation through the deployed-DCBM forward path.

The deployed-inference pipeline is three stages — the canonical names used
throughout this codebase, the JSONL `stage` field, and the paper:

    STAGE 1 — CLIP image embedding.
        Tensor: `clip_emb` shape (B, 512).
        Source: CLIP ViT-B/16 forward (`_clip_forward_factory` in pilot.py
        for corrupted; cached upstream embeddings for clean).
        Tracked metrics: `cosine_drift_mean`, `nearest_ref_drift_mean`.

    STAGE 2 — K-dim concept activation (post-softmax).
        Tensor: `concept_acts` shape (B, K), K=215 cifar10 / K=2048 imagenet.
        Math: `softmax(clip_emb @ centroids.T / ||centroids||², dim=1)`.
        Source: `propagate()` body below; mirrors upstream
        `utils/dcbm.py::preprocess_module` line 791-825.
        Headline metrics (per `decisions/2026-05-01_stage2_metric_set.md`):
        `argmax_shift_fraction`, `top5_jaccard_mean`, `spearman_rho_mean`.
        Sanity metrics: `per_concept_cosine_mean`, `delta_entropy_mean`,
        `kl_divergence_mean`.

    STAGE 3 — Linear-head prediction.
        Tensor: `logits` shape (B, num_classes); `preds = logits.argmax(-1)`.
        Math: `head( scaler.transform(concept_acts) )`.
        Source: `propagate()` body below; mirrors `CBM.predict` line 1216
        in upstream.
        Tracked metrics: `accuracy`, `prediction_shifts_vs_clean`, downstream mCE/RmCE.

`propagate` is the stages-2+3 primitive (given a stage-1 CLIP embedding).
`propagate_clean` materializes stages 2+3 over the upstream stage-1 cache
(`~/DCBM/data/Embeddings/images_<Dataset>_test_<model>.torch`) and persists
the output under `data/checkpoints/<dataset>_baseline/clean_propagation/`.
`propagate_corrupted_cell` runs one (corruption, severity) cell with a
caller-supplied CLIP forward + image loader, writes paired stage rows to
a `MetricsSink`, and optionally persists per-image (clean_top1, corrupt_top1)
argmax pairs for downstream centroid-jump / per-class flip analysis.
`compare_clean_vs_corrupted` aggregates a JSONL into a condition-keyed
summary dict.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from cbm_in_the_wild.metrics.clip_drift import cosine_drift, nearest_ref_drift
from cbm_in_the_wild.metrics.concept_stats import (
    argmax_shift_fraction,
    entropy_shift,
    kl_divergence,
    per_concept_cosine,
    softmax_entropy,
    spearman_rho,
    top_k_concept_overlap,
)
from cbm_in_the_wild.metrics.prediction import accuracy
from cbm_in_the_wild.metrics_sink import MetricsSink
from cbm_in_the_wild.paths import CHECKPOINT_DIR


# Canonical stage names — match the JSONL `stage` field.
STAGE_NAMES = ("clip_embedding", "concept_activation", "prediction")


# Per-dataset name conventions for upstream paths (see README
# §Canonical artifact paths — `cifar10` lowercase vs `ImageNet` PascalCase).
_UPSTREAM_DATASET_NAME = {"cifar10": "cifar10", "imagenet": "ImageNet"}
_DEFAULT_MODEL_TAG = "CLIP-ViT-B16"  # ViT-B/16 with slash stripped

_CLEAN_PROP_DIR = "clean_propagation"
_FILES = {
    "image_ids": "image_ids.pt",
    "labels": "labels.pt",
    "clip_emb": "clip_emb.pt",
    "concept_acts": "concept_act.pt",
    "logits": "logits.pt",
}


# -----------------------------------------------------------------------------
# Stage-2/3 primitive
# -----------------------------------------------------------------------------
def propagate(
    clip_emb: torch.Tensor,
    artifacts: dict,
    *,
    device: str | torch.device,
    recovery_fn=None,
) -> dict:
    """Stages 2+3 on a batch of CLIP embeddings.

    `recovery_fn` (optional): `(concept_acts: Tensor[N,K]) -> Tensor[N,K]`
    applied between bank softmax and scaler. Used by S2-reframe adapters to
    insert a closed-form K×K recovery layer at inference. No-op when None.

    Returns dict with `clip_emb` (cpu), `concept_acts` (cpu, *post-recovery*
    if `recovery_fn` set), `logits` (cpu), `preds` (cpu, int).
    """
    centroids = artifacts["centroids"].to(device)
    scaler = artifacts["scaler"]
    head = artifacts["linear_head"]

    norms_sq_t = (centroids ** 2).sum(dim=1, keepdim=True).t()  # (1, K)
    raw = (clip_emb.to(device).float() @ centroids.t()) / norms_sq_t
    concept_acts = torch.nn.functional.softmax(raw, dim=1).cpu()

    if recovery_fn is not None:
        concept_acts = recovery_fn(concept_acts)

    X = scaler.transform(concept_acts.numpy())
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = head(X_t).cpu()

    return {
        "clip_emb": clip_emb.detach().cpu(),
        "concept_acts": concept_acts,
        "logits": logits,
        "preds": logits.argmax(dim=-1),
    }


# -----------------------------------------------------------------------------
# Clean propagation — read upstream stage-1 cache, persist stages 2+3
# -----------------------------------------------------------------------------
def _resolve_dcbm_repo(dcbm_repo: Path | None) -> Path:
    if dcbm_repo is not None:
        return Path(dcbm_repo)
    import os
    return Path(os.environ.get("DCBM_REPO", Path.home() / "DCBM"))


def _upstream_emb_path(
    dataset: str,
    *,
    model_tag: str | None = None,
    dcbm_repo: Path | None = None,
) -> Path:
    repo = _resolve_dcbm_repo(dcbm_repo)
    name = _UPSTREAM_DATASET_NAME[dataset]
    tag = model_tag or _DEFAULT_MODEL_TAG
    return repo / "data" / "Embeddings" / f"images_{name}_test_{tag}.torch"


def _clean_prop_dir(
    dataset: str,
    *,
    baseline_name: str | None = None,
    ckpt_dir: Path | None = None,
) -> Path:
    base = Path(ckpt_dir) if ckpt_dir is not None else CHECKPOINT_DIR
    subdir = baseline_name or f"{dataset}_baseline"
    return base / subdir / _CLEAN_PROP_DIR


def _read_upstream_emb(path: Path) -> tuple[list[str], torch.Tensor, list[str]]:
    """Schema: {filename: [emb_tensor, class_name_str]}. Order = dict iteration."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    image_ids = list(blob.keys())
    embs = torch.stack([blob[k][0].float() for k in image_ids])
    class_names = [blob[k][1] for k in image_ids]
    return image_ids, embs, class_names


def propagate_clean(
    dataset: str,
    *,
    artifacts: dict,
    model_tag: str | None = None,
    baseline_name: str | None = None,
    dcbm_repo: Path | None = None,
    ckpt_dir: Path | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 1024,
    force: bool = False,
) -> dict:
    """Materialize stage-1 (read-through) + stage-2/3 (persisted) for clean test.

    Stage-1 source: `~/DCBM/data/Embeddings/images_<Dataset>_test_<model>.torch`.
    Stage-2/3 cached at `data/checkpoints/<baseline_name>/clean_propagation/`.
    `model_tag` selects the upstream embedding file (default CLIP-ViT-B16).
    `baseline_name` selects the checkpoint subdir (default `<dataset>_baseline`).
    `force=True` recomputes; default skips when all files exist.

    Returns:
        {"image_ids": list[str], "labels": Tensor (N,), "clip_emb": Tensor (N, D),
         "concept_acts": Tensor (N, K), "logits": Tensor (N, C), "preds": Tensor (N,)}
    """
    out_dir = _clean_prop_dir(dataset, baseline_name=baseline_name, ckpt_dir=ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / v for k, v in _FILES.items()}

    have_all = all(p.exists() for p in paths.values())
    if have_all and not force:
        logits = torch.load(paths["logits"])
        return {
            "image_ids": torch.load(paths["image_ids"], weights_only=False),
            "labels": torch.load(paths["labels"]),
            "clip_emb": torch.load(paths["clip_emb"]),
            "concept_acts": torch.load(paths["concept_acts"]),
            "logits": logits,
            "preds": logits.argmax(dim=-1),
        }

    src = _upstream_emb_path(dataset, model_tag=model_tag, dcbm_repo=dcbm_repo)
    if not src.exists():
        raise FileNotFoundError(f"upstream stage-1 cache missing: {src}")
    image_ids, clip_emb, class_names = _read_upstream_emb(src)

    sorted_classes = sorted(set(class_names))
    cls_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    labels = torch.tensor([cls_to_idx[c] for c in class_names], dtype=torch.long)

    concept_chunks, logits_chunks = [], []
    for start in range(0, clip_emb.shape[0], batch_size):
        chunk = clip_emb[start : start + batch_size]
        out = propagate(chunk, artifacts, device=device)
        concept_chunks.append(out["concept_acts"])
        logits_chunks.append(out["logits"])
    concept_acts = torch.cat(concept_chunks, dim=0)
    logits = torch.cat(logits_chunks, dim=0)

    torch.save(image_ids, paths["image_ids"])
    torch.save(labels, paths["labels"])
    torch.save(clip_emb, paths["clip_emb"])
    torch.save(concept_acts, paths["concept_acts"])
    torch.save(logits, paths["logits"])

    return {
        "image_ids": image_ids,
        "labels": labels,
        "clip_emb": clip_emb,
        "concept_acts": concept_acts,
        "logits": logits,
        "preds": logits.argmax(dim=-1),
    }


# -----------------------------------------------------------------------------
# Stratified sampler
# -----------------------------------------------------------------------------
def stratified_sampler(
    image_ids: list[str],
    labels: torch.Tensor,
    *,
    n_per_class: int,
    n_classes: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Pick `n_per_class * (n_classes or num_classes)` images stratified by label.

    If `n_classes` is provided and < total class count, randomly subsample
    classes first (used by ImageNet 10/class × 100). Returns (image_ids, indices)
    where indices are positions into the input list.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        by_class[lab].append(i)

    all_classes = sorted(by_class.keys())
    if n_classes is not None and n_classes < len(all_classes):
        chosen_classes = rng.sample(all_classes, n_classes)
    else:
        chosen_classes = all_classes

    picked: list[int] = []
    for c in chosen_classes:
        pool = by_class[c]
        if len(pool) < n_per_class:
            raise ValueError(
                f"class {c} has only {len(pool)} samples, need {n_per_class}"
            )
        picked.extend(rng.sample(pool, n_per_class))

    picked.sort()
    return [image_ids[i] for i in picked], picked


# -----------------------------------------------------------------------------
# Corrupted-cell forward + paired metrics
# -----------------------------------------------------------------------------
def propagate_corrupted_cell(
    *,
    dataset: str,
    corruption: str,
    severity: int,
    image_ids: list[str],
    clean_subset: dict,
    artifacts: dict,
    clip_forward: Callable[[np.ndarray], torch.Tensor],
    corrupt_fn: Callable[[list[str], str, int], np.ndarray],
    sink: MetricsSink,
    device: str | torch.device,
    per_image_dir: Path | None = None,
    dump_stage_tensors: bool = False,
    batch_size: int = 512,
) -> dict:
    """Run one (corruption, severity) cell, paired with clean_subset.

    `clean_subset`: dict with `clip_emb`, `concept_acts`, `preds`, `labels` —
    each a Tensor of length `len(image_ids)`, ordered to match.
    `corrupt_fn(image_ids, corruption, severity) -> uint8 (N, H, W, 3) NHWC`.
    `clip_forward(uint8 NHWC) -> Tensor (N, D) on cpu`.

    `batch_size` chunks the corrupt+forward loop. Without chunking, ImageNet
    50 k cells materialise a 7.5 GB uint8 stack + a 30 GB GPU input batch and
    OOM at --mem=64G; chunking caps both at the per-batch footprint. CIFAR-10
    is small enough that batch_size doesn't matter operationally.

    `dump_stage_tensors=True` adds per-image stage-1 (`clip_emb`, N×512) and
    stage-2 (`concept_acts`, N×K) full tensors to the per-image dump alongside
    the always-on argmax/pred payload. Apply selectively — adds ≈ 28 MB/cell
    on cifar10, ≈ 500 MB/cell on imagenet. Required by §11 cluster-smear and
    §12 rank-rank scatter and the per-concept robustness leaderboard.

    Writes three stage rows (`clip_embedding`, `concept_activation`,
    `prediction`) to sink. Returns per-image preds + clip_emb (for
    cross-severity perturbation aggregation by the caller).
    """
    clip_emb_chunks: list[torch.Tensor] = []
    for start in range(0, len(image_ids), batch_size):
        chunk_ids = image_ids[start : start + batch_size]
        chunk_imgs = corrupt_fn(chunk_ids, corruption, severity)
        chunk_emb = clip_forward(chunk_imgs)
        clip_emb_chunks.append(chunk_emb)
        del chunk_imgs
    clip_emb = torch.cat(clip_emb_chunks, dim=0)
    del clip_emb_chunks
    out = propagate(clip_emb, artifacts, device=device)

    co_clip = clean_subset["clip_emb"]
    co_concept = clean_subset["concept_acts"]
    co_preds = clean_subset["preds"]
    labels = clean_subset["labels"]

    cd = cosine_drift(co_clip, out["clip_emb"])
    nrd = nearest_ref_drift(out["clip_emb"], co_clip)
    sink.write("clip_embedding", {
        "dataset": dataset,
        "corruption": corruption,
        "severity": severity,
        "n": int(co_clip.shape[0]),
        "cosine_drift_mean": float(cd.mean()),
        "cosine_drift_std": float(cd.std(unbiased=False)),
        "nearest_ref_drift_mean": nrd["mean"],
    })

    pcc = per_concept_cosine(co_concept, out["concept_acts"])
    h_clean = softmax_entropy(co_concept)
    es = entropy_shift(co_concept, out["concept_acts"])
    tk = top_k_concept_overlap(co_concept, out["concept_acts"], k=5)
    asf = argmax_shift_fraction(co_concept, out["concept_acts"])
    kl = kl_divergence(co_concept, out["concept_acts"])
    sp = spearman_rho(co_concept, out["concept_acts"])
    sink.write("concept_activation", {
        "dataset": dataset,
        "corruption": corruption,
        "severity": severity,
        "per_concept_cosine_mean": float(pcc.mean()),
        "softmax_entropy_clean_mean": float(h_clean.mean()),
        "delta_entropy_mean": float(es["delta_entropy"].mean()),
        "top5_jaccard_mean": float(tk.mean()),
        "argmax_shift_fraction": asf,
        "kl_divergence_mean": float(kl.mean()),
        "spearman_rho_mean": float(sp.mean()),
    })

    acc = accuracy(out["preds"], labels)
    pred_shifts = int((out["preds"] != co_preds).sum().item())
    sink.write("prediction", {
        "dataset": dataset,
        "corruption": corruption,
        "severity": severity,
        "accuracy": acc,
        "prediction_shifts_vs_clean": pred_shifts,
    })

    # Optional per-image argmax dump for centroid-jump + per-class flip analysis.
    # `dump_stage_tensors=True` enriches the same .pt with full stage-1/2 tensors.
    if per_image_dir is not None:
        per_image_dir = Path(per_image_dir)
        per_image_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_ids": list(image_ids),
            "labels": labels.cpu(),
            "clean_top1_concept": co_concept.argmax(dim=-1).cpu(),
            "corrupt_top1_concept": out["concept_acts"].argmax(dim=-1).cpu(),
            "clean_pred": co_preds.cpu(),
            "corrupt_pred": out["preds"].cpu(),
        }
        if dump_stage_tensors:
            payload["clip_emb"] = out["clip_emb"].cpu()
            payload["concept_acts"] = out["concept_acts"].cpu()
        torch.save(payload, per_image_dir / f"{corruption}_s{severity}.pt")

    return {"preds": out["preds"], "clip_emb": out["clip_emb"]}


# -----------------------------------------------------------------------------
# Aggregator over a stage_metrics.jsonl
# -----------------------------------------------------------------------------
def compare_clean_vs_corrupted(jsonl_path: Path) -> dict:
    """Aggregate a `stage_metrics.jsonl` produced by `propagate_corrupted_cell`.

    Returns a `(dataset, corruption, severity)`-keyed dict of merged stage
    fields (clip_embedding + concept_activation + prediction rows folded
    into one record per cell). Used by pilot runners to write summary.json.
    """
    by_cell: dict[tuple[str, str, int], dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row.get("dataset"), row.get("corruption"), row.get("severity"))
            if None in key:
                continue
            cell = by_cell.setdefault(key, {})
            for k, v in row.items():
                if k in ("stage", "dataset", "corruption", "severity"):
                    continue
                cell[k] = v
    return {
        f"{ds}|{c}|{s}": v
        for (ds, c, s), v in sorted(by_cell.items())
    }
