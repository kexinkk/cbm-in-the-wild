"""Corrupted test-split CLIP-embedding cells for the LF-CBM ablation.

ImageNet: reuse the 50 full-test cells already cached by
experiments/imagenet_c_propagation/<fam>_s<sev>/per_image/ (clip_emb key).

CIFAR-10: those cells stored no clip_emb, so a one-off 50-cell GPU array
regenerates them through the *same* harness (setup_for_cells, seed 42 →
identical corruption realizations as cifar10_c_propagation). CIFAR-10-C is
bulk .npy and the 6 sensor corruptions are synthesized in-memory — no loose
raw tree is iterated and nothing is tarred.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cbm_in_the_wild.paths import REPO_ROOT

IMAGENET_PROP_ROOT = REPO_ROOT / "experiments" / "imagenet_c_propagation"


def cell_path(cfg: dict, exp_dir: Path, corruption: str, severity: int) -> Path:
    if cfg["dataset"] == "imagenet":
        d = IMAGENET_PROP_ROOT / f"{corruption}_s{severity}" / "per_image"
        return d / f"{corruption}_s{severity}.pt"
    # Optional shared cache: a repo-relative (or absolute) dir of corrupted CLIP
    # embeddings z(x), which are backbone-only and family-agnostic.
    ctd = cfg.get("corrupted_test_dir")
    if ctd:
        base = Path(ctd)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / f"{corruption}_s{severity}.pt"
    return exp_dir / "corrupted_test" / f"{corruption}_s{severity}.pt"


def load_cell(cfg: dict, exp_dir: Path, corruption: str, severity: int) -> dict:
    """→ {"clip_emb": [N,512] f32, "image_ids": list[str]}"""
    blob = torch.load(cell_path(cfg, exp_dir, corruption, severity), map_location="cpu")
    return {"clip_emb": blob["clip_emb"].float(), "image_ids": list(blob["image_ids"])}


def generate_cifar_cell(cfg: dict, exp_dir: Path, task_id: int) -> None:
    """One (corruption, severity) cell: corrupt full CIFAR-10 test, embed."""
    from cbm_in_the_wild.corruption_eval import resolve_corruptions, setup_for_cells

    corruptions = resolve_corruptions(cfg["corruption_set"])
    severities = tuple(cfg["severities"])
    corruption = corruptions[task_id // len(severities)]
    severity = severities[task_id % len(severities)]

    out = cell_path(cfg, exp_dir, corruption, severity)
    if out.exists():
        print(f"[cell] {corruption}_s{severity} exists, skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    harness_cfg = {
        "dataset": "cifar10",
        "seed": 42,  # match cifar10_c_propagation corruption realizations
        "corruption_set": cfg["corruption_set"],
        "severities": list(severities),
        "device": "cuda",
    }
    ctx = setup_for_cells(harness_cfg, full_test=True)

    chunks = []
    for start in range(0, len(ctx.eval_ids), 512):
        ids = ctx.eval_ids[start : start + 512]
        imgs = ctx.corrupt_fn(ids, corruption, severity)
        chunks.append(ctx.clip_forward(imgs))
        del imgs
    clip_emb = torch.cat(chunks).cpu().float()

    torch.save(
        {
            "clip_emb": clip_emb,
            "image_ids": list(ctx.eval_ids),
            "labels": ctx.clean_eval["labels"].cpu(),
        },
        out,
    )
    print(f"[cell] wrote {out} n={len(ctx.eval_ids)}")


