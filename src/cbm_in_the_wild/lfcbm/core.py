"""Shared LF-CBM machinery: embedding IO, artifacts, deployed forward path."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from cbm_in_the_wild.paths import REPO_ROOT

LFCBM_REPO = Path(os.environ.get("LFCBM_REPO", Path.home() / "Label-free-CBM"))
DCBM_REPO = Path(os.environ.get("DCBM_REPO", REPO_ROOT.parent / "DCBM"))

_DS_CASING = {"cifar10": "cifar10", "imagenet": "ImageNet", "cub": "CUB", "derm7pt": "derm7pt", "skincon": "skincon"}


def add_lfcbm_to_path() -> None:
    """Make upstream glm_saga / similarity importable without editing the fork."""
    p = str(LFCBM_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


def emb_path(dataset: str, split: str) -> Path:
    ds = _DS_CASING[dataset]
    return DCBM_REPO / "data" / "Embeddings" / f"images_{ds}_{split}_CLIP-ViT-B16.torch"


def load_emb_split(dataset: str, split: str, cache: bool = True):
    """Load a cached clean-embedding dict in deterministic (sorted-id) order.

    Returns (ids, X float32 [N,512], y int64 [N], classes sorted list).
    Class index = position in sorted(unique label strings) — self-consistent
    across splits (wnid order for ImageNet, class-name order for CIFAR).

    First call stacks the upstream per-image dict (slow for the 1.15M-entry
    ImageNet train file) and caches the result in the workspace; later calls
    load the stack in seconds.
    """
    from cbm_in_the_wild.paths import DATA_DIR

    stack = DATA_DIR / "lfcbm" / dataset / f"stack_{split}.pt"
    if cache and stack.exists():
        b = torch.load(stack, map_location="cpu")
        return b["ids"], b["X"].float(), b["y"], b["classes"]
    blob = torch.load(emb_path(dataset, split), map_location="cpu")
    ids = sorted(blob.keys())
    X = torch.stack([blob[i][0].float() for i in ids])
    labels = [blob[i][1] for i in ids]
    classes = sorted(set(labels))
    cls_to_idx = {c: k for k, c in enumerate(classes)}
    y = torch.tensor([cls_to_idx[l] for l in labels], dtype=torch.long)
    if cache:
        stack.parent.mkdir(parents=True, exist_ok=True)
        tmp = stack.with_suffix(".tmp")
        torch.save({"ids": ids, "X": X.half(), "y": y, "classes": classes}, tmp)
        tmp.rename(stack)
    return ids, X, y, classes


# --- artifacts ---------------------------------------------------------------

def save_stage_a(model_dir: Path, W_c, proj_mean, proj_std, concepts, meta):
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(W_c.cpu(), model_dir / "W_c.pt")
    torch.save(proj_mean.cpu(), model_dir / "proj_mean.pt")
    torch.save(proj_std.cpu(), model_dir / "proj_std.pt")
    (model_dir / "concepts.txt").write_text("\n".join(concepts))
    (model_dir / "stage_a_meta.json").write_text(json.dumps(meta, indent=2))


def save_head(model_dir: Path, W_g, b_g, meta):
    torch.save(W_g.cpu(), model_dir / "W_g.pt")
    torch.save(b_g.cpu(), model_dir / "b_g.pt")
    (model_dir / "head_meta.json").write_text(json.dumps(meta, indent=2))


def load_artifacts(model_dir: Path, device: str = "cpu") -> dict:
    art = {
        "W_c": torch.load(model_dir / "W_c.pt", map_location=device),
        "proj_mean": torch.load(model_dir / "proj_mean.pt", map_location=device),
        "proj_std": torch.load(model_dir / "proj_std.pt", map_location=device),
        "concepts": (model_dir / "concepts.txt").read_text().split("\n"),
    }
    if (model_dir / "W_g.pt").exists():
        art["W_g"] = torch.load(model_dir / "W_g.pt", map_location=device)
        art["b_g"] = torch.load(model_dir / "b_g.pt", map_location=device)
    return art


# --- deployed forward path ---------------------------------------------------

def raw_acts_from_emb(z: torch.Tensor, art: dict) -> torch.Tensor:
    """z [N,512] → raw concept projection scores c = W_c z (pre-standardization).

    This is the space `d_k` is measured in — the DCBM-comparable analog of
    DCBM's raw pre-softmax score z·Cᵀ/‖C‖². W_c rows are near-unit-norm on
    ImageNet (‖W_c,k‖ p90/p10 ≈ 1.1, σ_k p90/p10 ≈ 1.3), so raw scores are
    already cross-concept comparable and no per-concept σ reweighting is
    applied. NOT the head-input space (the head reads `std_acts_from_emb`).
    """
    return z.float() @ art["W_c"].t()


def std_acts_from_emb(z: torch.Tensor, art: dict) -> torch.Tensor:
    """z [N,512] → standardized concept activations (what the head reads)."""
    c = z.float() @ art["W_c"].t()
    return (c - art["proj_mean"]) / art["proj_std"]


def logits_from_std(a: torch.Tensor, art: dict, keep_idx=None) -> torch.Tensor:
    W, b = art["W_g"], art["b_g"]
    if keep_idx is not None:
        return a[:, keep_idx] @ W[:, keep_idx].t() + b
    return a @ W.t() + b


# --- CLIP text encoding ------------------------------------------------------

def encode_concept_texts(concepts: list[str], clip_name: str, device: str) -> torch.Tensor:
    import clip

    model, _ = clip.load(clip_name, device=device)
    tok = clip.tokenize(concepts).to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(tok), 512):
            feats.append(model.encode_text(tok[i : i + 512]).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)
