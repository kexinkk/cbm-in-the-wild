"""Phase-2 corruption evaluation orchestrator.

Sits between `pipeline.py` (pure DCBM forward math) and per-experiment
runners. Knows about CLIP loading, image loading, corruption application,
JSONL writing, and (corruption, severity) cell orchestration.

Two entry points:

- `run_pilot(cfg, *, out_dir)` — Tier-1 multi-cell-on-one-process driver
  (50 cells in one Python process, one JSONL, cross-severity flip-rate
  aggregation in the same run). Used by `experiments/pilot_*/run.py`.
- `run_one_cell(ctx, corruption, severity, *, cell_dir)` — Tier-2 array
  driver (one cell per SLURM array task). Used by
  `experiments/<dataset>_c_propagation/run_cell.py`. Build `ctx` once via
  `setup_for_cells(cfg, *, full_test=True)`.

Both share the setup helpers and corruptor factories below.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from cbm_in_the_wild.corruptions import (
    CIFAR10C_SELECTED,
    IMAGENET_C_SELECTED,
    SENSOR_TRANSFORMS,
    imagenet_c_severity_dir,
    load_cifar10c,
)
def _severity_shift_rate(seq):
    """Adjacent-disagreement rate over a severity-indexed sequence.

    Continuum-axis primitive (severity 1→5): inlined here to keep the
    perturbation namespace clean of corruption-context callers.
    """
    if len(seq) < 2:
        return 0.0
    return sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b) / (len(seq) - 1)
from cbm_in_the_wild.metrics_sink import MetricsSink
from cbm_in_the_wild.paths import (
    CHECKPOINT_DIR,
    cifar10_dir,
    imagenet_dir,
)
from cbm_in_the_wild.persistence import load_cbm_artifacts
from cbm_in_the_wild.pipeline import (
    compare_clean_vs_corrupted,
    propagate_clean,
    propagate_corrupted_cell,
    stratified_sampler,
)
from cbm_in_the_wild.runs import capture_context, write_summary
from cbm_in_the_wild.seeding import set_all_seeds


SEVERITIES = (1, 2, 3, 4, 5)


# -----------------------------------------------------------------------------
# CLIP forward + sensor application helpers
# -----------------------------------------------------------------------------
_MODEL_TAG_TO_CLIP_ARCH = {
    "CLIP-ViT-B16": "ViT-B/16",
    "CLIP-RN50": "RN50",
}


def _load_clip(device: str, clip_arch: str = "ViT-B/16"):
    import clip  # local import — heavy deps
    model, preprocess = clip.load(clip_arch, device=device, jit=False)
    model.eval()
    return model, preprocess


def _clip_forward_factory(model, preprocess, device: str):
    def forward(images_uint8_NHWC: np.ndarray) -> torch.Tensor:
        pil = [Image.fromarray(img) for img in images_uint8_NHWC]
        batch = torch.stack([preprocess(im) for im in pil]).to(device)
        with torch.no_grad():
            return model.encode_image(batch).float().cpu()
    return forward


def _apply_sensor(img: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    cls = SENSOR_TRANSFORMS[name]
    t = torch.from_numpy(img).permute(2, 0, 1)
    g = torch.Generator().manual_seed(seed)
    return cls(severity=severity, generator=g)(t).permute(1, 2, 0).numpy().astype(np.uint8)


# -----------------------------------------------------------------------------
# Per-dataset corruptor factories
# -----------------------------------------------------------------------------
def _cifar10_id_to_row(image_id: str) -> int:
    return int(Path(image_id).stem)


def _cifar10_clean_image(image_id: str, class_name: str) -> np.ndarray:
    p = cifar10_dir() / "test" / class_name / image_id
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def _imagenet_clean_image(image_id: str, wnid: str) -> np.ndarray:
    """Return 224×224 CenterCrop'd uint8 RGB — matches Hendrycks ImageNet-C."""
    from torchvision import transforms as T
    p = imagenet_dir() / "test" / wnid / image_id
    cc = T.Compose([T.Resize(256), T.CenterCrop(224)])
    with Image.open(p) as im:
        return np.asarray(cc(im.convert("RGB")))


def _build_cifar10_corrupt_fn(class_names_by_id: dict[str, str]):
    """One callable for both Hendrycks and sensor corruptions on CIFAR-10."""

    def corrupt(image_ids: list[str], corruption: str, severity: int) -> np.ndarray:
        if corruption in CIFAR10C_SELECTED:
            imgs, _ = load_cifar10c(corruption, severity)
            rows = [_cifar10_id_to_row(i) for i in image_ids]
            return np.stack([np.asarray(imgs[r]) for r in rows])
        # sensor: load clean, apply transform per-image with deterministic seed
        out = []
        for k, image_id in enumerate(image_ids):
            clean = _cifar10_clean_image(image_id, class_names_by_id[image_id])
            seed = hash((image_id, corruption, severity)) & 0xFFFFFFFF
            out.append(_apply_sensor(clean, corruption, severity, seed))
        return np.stack(out)

    return corrupt


def _build_imagenet_corrupt_fn(class_names_by_id: dict[str, str]):
    """One callable for both Hendrycks and sensor corruptions on ImageNet."""
    from torchvision import transforms as T
    cc = T.Compose([T.Resize(256), T.CenterCrop(224)])

    def corrupt(image_ids: list[str], corruption: str, severity: int) -> np.ndarray:
        if corruption in IMAGENET_C_SELECTED:
            sev_dir = imagenet_c_severity_dir(corruption, severity)
            out = []
            for image_id in image_ids:
                wnid = class_names_by_id[image_id]
                p = sev_dir / wnid / image_id
                with Image.open(p) as im:
                    out.append(np.asarray(cc(im.convert("RGB"))))
            return np.stack(out)
        out = []
        for image_id in image_ids:
            clean = _imagenet_clean_image(image_id, class_names_by_id[image_id])
            seed = hash((image_id, corruption, severity)) & 0xFFFFFFFF
            out.append(_apply_sensor(clean, corruption, severity, seed))
        return np.stack(out)

    return corrupt


# -----------------------------------------------------------------------------
# Cell-mode setup
# -----------------------------------------------------------------------------
def resolve_corruptions(corruption_set: str) -> list[str]:
    """Map cfg.corruption_set → ordered list of corruption names.

    Order is stable across processes — Tier-2 array tasks rely on
    `corruptions[task_id // 5]` to map SLURM_ARRAY_TASK_ID to a cell.
    """
    if corruption_set == "combined":
        return list(CIFAR10C_SELECTED) + list(SENSOR_TRANSFORMS)
    if corruption_set == "hendrycks":
        return list(IMAGENET_C_SELECTED)
    if corruption_set == "sensor":
        return list(SENSOR_TRANSFORMS)
    raise ValueError(
        f"corruption_set must be 'combined' | 'hendrycks' | 'sensor', got {corruption_set!r}"
    )


@dataclass
class CellContext:
    """Setup state shared by every (corruption, severity) cell in one process."""
    dataset: str
    severities: tuple[int, ...]
    corruptions: list[str]
    artifacts: dict
    clean_eval: dict        # dict with clip_emb/concept_acts/logits/preds/labels
    eval_ids: list[str]     # image ids paired with clean_eval rows
    clip_forward: Callable
    corrupt_fn: Callable
    device: str
    manifest: dict
    seed: int


def _upstream_emb_path(dataset: str, model_tag: str | None = None) -> Path:
    from cbm_in_the_wild.pipeline import _upstream_emb_path as _p
    return _p(dataset, model_tag=model_tag)


def _class_names_cache_path(dataset: str, baseline_name: str | None = None) -> Path:
    subdir = baseline_name or f"{dataset}_baseline"
    return CHECKPOINT_DIR / subdir / "clean_propagation" / "class_names.json"


def _load_class_names(
    dataset: str,
    image_ids: list[str],
    *,
    model_tag: str | None = None,
    baseline_name: str | None = None,
) -> dict[str, str]:
    """Map each image_id → its class string (cifar class name / imagenet wnid).

    Cached to `clean_propagation/class_names.json` after first compute, so
    Tier-2 array tasks pay the upstream-embedding load cost once per dataset
    rather than 50 times.
    """
    cache_path = _class_names_cache_path(dataset, baseline_name)
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if all(i in cached for i in image_ids):
            return {i: cached[i] for i in image_ids}

    src = _upstream_emb_path(dataset, model_tag)
    blob = torch.load(str(src), map_location="cpu", weights_only=False)
    full = {k: blob[k][1] for k in blob}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(full, f)

    return {i: full[i] for i in image_ids}


def setup_for_cells(cfg: dict, *, full_test: bool) -> CellContext:
    """Materialize everything needed to run any cell: artifacts, clean
    propagation cache (or stratified subset), CLIP forward, corruptor factory.

    `full_test=False` (Tier-1) → stratified n_per_class subset.
    `full_test=True`  (Tier-2) → entire clean test set as eval pool.

    cfg keys for model selection (optional, default = ViT-B/16 baseline):
        model_name:    "CLIP-ViT-B16" | "CLIP-RN50"
        baseline_name: checkpoint subdir under data/checkpoints/ (e.g.
                       "cifar10_rn50_baseline"); defaults to "<dataset>_baseline".
    """
    seed = int(cfg.get("seed", 42))
    set_all_seeds(seed)

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("GPU required for corruption evaluation")

    dataset = cfg["dataset"]
    severities = tuple(cfg.get("severities", SEVERITIES))
    corruptions = resolve_corruptions(cfg["corruption_set"])

    model_tag = cfg.get("model_name", "CLIP-ViT-B16")
    baseline_name = cfg.get("baseline_name", f"{dataset}_baseline")
    clip_arch = _MODEL_TAG_TO_CLIP_ARCH.get(model_tag, "ViT-B/16")

    art_dir = CHECKPOINT_DIR / baseline_name
    artifacts = load_cbm_artifacts(art_dir, device=device)
    manifest = artifacts["manifest"]
    print(f"[setup] {model_tag} artifacts K={manifest['n_clusters']} "
          f"D={manifest['clip_dim']} C={manifest['num_classes']} "
          f"test_acc={manifest.get('test_accuracy', '?')}")

    t0 = time.time()
    clean = propagate_clean(
        dataset, artifacts=artifacts, model_tag=model_tag,
        baseline_name=baseline_name, device=device,
    )
    print(f"[clean] N={len(clean['image_ids'])} took {time.time() - t0:.1f}s")

    if full_test:
        eval_ids = list(clean["image_ids"])
        clean_eval = {
            "clip_emb": clean["clip_emb"],
            "concept_acts": clean["concept_acts"],
            "logits": clean["logits"],
            "preds": clean["preds"],
            "labels": clean["labels"],
        }
        print(f"[eval] full test set, n={len(eval_ids)}")
    else:
        picked_ids, picked_idx = stratified_sampler(
            clean["image_ids"], clean["labels"],
            n_per_class=int(cfg["n_per_class"]),
            n_classes=cfg.get("n_classes"),
            seed=seed,
        )
        idx_t = torch.tensor(picked_idx, dtype=torch.long)
        eval_ids = picked_ids
        clean_eval = {
            "clip_emb": clean["clip_emb"][idx_t],
            "concept_acts": clean["concept_acts"][idx_t],
            "logits": clean["logits"][idx_t],
            "preds": clean["preds"][idx_t],
            "labels": clean["labels"][idx_t],
        }
        print(f"[eval] stratified n={len(eval_ids)}, seed={seed}")

    class_names_by_id = _load_class_names(
        dataset, eval_ids, model_tag=model_tag, baseline_name=baseline_name,
    )

    clip_model, preprocess = _load_clip(device, clip_arch=clip_arch)
    clip_forward = _clip_forward_factory(clip_model, preprocess, device)
    if dataset == "cifar10":
        corrupt_fn = _build_cifar10_corrupt_fn(class_names_by_id)
    elif dataset == "imagenet":
        corrupt_fn = _build_imagenet_corrupt_fn(class_names_by_id)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    return CellContext(
        dataset=dataset,
        severities=severities,
        corruptions=corruptions,
        artifacts=artifacts,
        clean_eval=clean_eval,
        eval_ids=eval_ids,
        clip_forward=clip_forward,
        corrupt_fn=corrupt_fn,
        device=device,
        manifest=manifest,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# Tier-2 single-cell driver
# -----------------------------------------------------------------------------
def run_one_cell(
    ctx: CellContext,
    corruption: str,
    severity: int,
    *,
    cell_dir: Path,
    cfg_for_summary: dict | None = None,
    dump_stage_tensors: bool = False,
) -> None:
    """Run one (corruption, severity) cell and write per-cell artifacts.

    Outputs in `cell_dir`:
        stage_metrics.jsonl   — clip_embedding + concept_activation + prediction rows
        per_image/<corruption>_s<severity>.pt  — argmax + pred dumps
                                                 (+ clip_emb + concept_acts when
                                                  `dump_stage_tensors=True`)
        summary.json          — config, context, timing, by_cell metrics
    """
    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cell_dir / "stage_metrics.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    per_image_dir = cell_dir / "per_image"

    t0 = time.time()
    with MetricsSink(jsonl_path) as sink:
        sink.write("clean_baseline", {
            "dataset": ctx.dataset,
            "n": len(ctx.eval_ids),
            "clean_eval_accuracy": float(
                (ctx.clean_eval["preds"] == ctx.clean_eval["labels"]).float().mean()
            ),
            "manifest": ctx.manifest,
        })
        propagate_corrupted_cell(
            dataset=ctx.dataset,
            corruption=corruption,
            severity=severity,
            image_ids=ctx.eval_ids,
            clean_subset=ctx.clean_eval,
            artifacts=ctx.artifacts,
            clip_forward=ctx.clip_forward,
            corrupt_fn=ctx.corrupt_fn,
            sink=sink,
            device=ctx.device,
            per_image_dir=per_image_dir,
            dump_stage_tensors=dump_stage_tensors,
        )
    wall = time.time() - t0
    print(f"[cell] {corruption} sev={severity} done in {wall:.1f}s")

    by_cell = compare_clean_vs_corrupted(jsonl_path)
    write_summary(
        cell_dir / "summary.json",
        config=cfg_for_summary or {
            "dataset": ctx.dataset,
            "corruption": corruption,
            "severity": severity,
            "seed": ctx.seed,
        },
        metrics={"by_cell": by_cell, "n_cells": 1},
        timing_seconds={"forward": wall},
        context=capture_context(Path(__file__).resolve().parents[2]),
        extra={
            "n_eval": len(ctx.eval_ids),
            "corruption": corruption,
            "severity": severity,
        },
    )
    print(f"[summary] wrote {cell_dir / 'summary.json'}")


# -----------------------------------------------------------------------------
# Tier-1 multi-cell driver (backward compat — used by experiments/pilot_*/run.py)
# -----------------------------------------------------------------------------
def run_pilot(cfg: dict, *, out_dir: Path) -> None:
    """Run one Tier-1 pilot end-to-end (50 cells in one process).

    Required cfg keys: `dataset`, `seed`, `n_per_class`, `corruption_set`.
    Optional: `n_classes` (ImageNet stratified), `severities`, `device`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = setup_for_cells(cfg, full_test=False)
    severities = ctx.severities
    corruptions = ctx.corruptions

    jsonl_path = out_dir / "stage_metrics.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    per_image_dir = out_dir / "per_image"

    per_image_preds: dict[str, dict[int, torch.Tensor]] = {
        c: {} for c in corruptions
    }
    per_image_clip: dict[str, dict[int, torch.Tensor]] = {
        c: {} for c in corruptions
    }
    t1 = time.time()
    n_cells = 0
    with MetricsSink(jsonl_path) as sink:
        sink.write("clean_baseline", {
            "dataset": ctx.dataset,
            "n": len(ctx.eval_ids),
            "clean_subset_accuracy": float(
                (ctx.clean_eval["preds"] == ctx.clean_eval["labels"]).float().mean()
            ),
            "manifest": ctx.manifest,
        })
        for corruption in corruptions:
            for sev in severities:
                cell = propagate_corrupted_cell(
                    dataset=ctx.dataset,
                    corruption=corruption,
                    severity=sev,
                    image_ids=ctx.eval_ids,
                    clean_subset=ctx.clean_eval,
                    artifacts=ctx.artifacts,
                    clip_forward=ctx.clip_forward,
                    corrupt_fn=ctx.corrupt_fn,
                    sink=sink,
                    device=ctx.device,
                    per_image_dir=per_image_dir,
                )
                per_image_preds[corruption][sev] = cell["preds"]
                per_image_clip[corruption][sev] = cell["clip_emb"]
                n_cells += 1
                print(f"[cell] {corruption} sev={sev} done")

        n_subset = len(ctx.eval_ids)
        for corruption in corruptions:
            for k in range(n_subset):
                preds_seq = [int(per_image_preds[corruption][s][k].item()) for s in severities]
                clip_seq = [per_image_clip[corruption][s][k] for s in severities]
                pred_ssr = _severity_shift_rate(preds_seq)
                clip_argmax_ssr = _severity_shift_rate(
                    [int(t.argmax().item()) for t in clip_seq],
                )
                sink.write("severity_sequence", {
                    "dataset": ctx.dataset,
                    "corruption": corruption,
                    "image_idx": k,
                    "image_id": ctx.eval_ids[k],
                    "severity_shift_rate_predictions": pred_ssr,
                    "stage_wise_severity_shift_rate_clip_argmax": clip_argmax_ssr,
                })

    wall = time.time() - t1
    print(f"[forward] {n_cells} cells in {wall:.1f}s")

    summary_metrics = compare_clean_vs_corrupted(jsonl_path)
    write_summary(
        out_dir / "summary.json",
        config=cfg,
        metrics={"by_cell": summary_metrics, "n_cells": n_cells},
        timing_seconds={"forward": wall},
        context=capture_context(Path(__file__).resolve().parents[2]),
        extra={
            "n_subset": len(ctx.eval_ids),
            "corruptions": corruptions,
            "severities": list(severities),
        },
    )
    print(f"[summary] wrote {out_dir / 'summary.json'}")
