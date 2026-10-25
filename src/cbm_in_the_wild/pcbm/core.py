"""Shared PCBM machinery: embedding IO, concept-bank artifacts, forward path.

Self-contained (does NOT import from lfcbm) so the two genericity points stay
decoupled — tweaking one never perturbs the other's committed results. The
generic embedding IO (`load_emb_split`, `emb_path`, `encode_concept_texts`) is
copied verbatim from lfcbm/core.py; only the concept-bank layer differs.

Bank artifact schema (per model dir, one dir per concept source):
    cavs.pt        [K,512]  concept vectors (SVM coef_ for cav; unit-norm text
                            features for mm)
    intercepts.pt  [K]      SVM intercept_ (0 for mm)
    norms.pt       [K]      ‖coef_‖ (1 for mm) — the PCBM margin normaliser
    concepts.txt            K concept names, order-aligned with the rows above
    proj_mean.pt   [1,K]    train-set mean of raw projection (standardiser)
    proj_std.pt    [1,K]    train-set std  of raw projection
    W_g.pt / b_g.pt         sparse head (added by train stage B)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from cbm_in_the_wild.paths import REPO_ROOT

DCBM_REPO = Path(os.environ.get("DCBM_REPO", REPO_ROOT.parent / "DCBM"))

_DS_CASING = {"cifar10": "cifar10", "imagenet": "ImageNet", "cub": "CUB",
              "skincon": "skincon", "derm7pt": "derm7pt"}   # skin sets added 2026-10-07 (CAV PCBM)


def emb_path(dataset: str, split: str) -> Path:
    ds = _DS_CASING[dataset]
    return DCBM_REPO / "data" / "Embeddings" / f"images_{ds}_{split}_CLIP-ViT-B16.torch"


def load_emb_split(dataset: str, split: str, cache: bool = True):
    """Load a cached clean-embedding dict in deterministic (sorted-id) order.

    Returns (ids, X float32 [N,512], y int64 [N], classes sorted list). Class
    index = position in sorted(unique label strings) — self-consistent across
    splits. Reuses the same workspace stack cache the LF-CBM point built (the
    embeddings are identical: same CLIP ViT-B/16 files), so no re-stacking of
    the 1.15M-entry ImageNet train file.
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

def save_bank(model_dir: Path, cavs, intercepts, norms, concepts, meta) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cavs.cpu().float(), model_dir / "cavs.pt")
    torch.save(intercepts.cpu().float(), model_dir / "intercepts.pt")
    torch.save(norms.cpu().float(), model_dir / "norms.pt")
    (model_dir / "concepts.txt").write_text("\n".join(concepts))
    (model_dir / "bank_meta.json").write_text(json.dumps(meta, indent=2))


def save_proj_stats(model_dir: Path, proj_mean, proj_std) -> None:
    torch.save(proj_mean.cpu(), model_dir / "proj_mean.pt")
    torch.save(proj_std.cpu(), model_dir / "proj_std.pt")


def save_head(model_dir: Path, W_g, b_g, meta) -> None:
    torch.save(W_g.cpu(), model_dir / "W_g.pt")
    torch.save(b_g.cpu(), model_dir / "b_g.pt")
    (model_dir / "head_meta.json").write_text(json.dumps(meta, indent=2))


def load_artifacts(model_dir: Path, device: str = "cpu") -> dict:
    cavs = torch.load(model_dir / "cavs.pt", map_location=device)
    art = {
        "cavs": cavs,
        "intercepts": torch.load(model_dir / "intercepts.pt", map_location=device),
        "norms": torch.load(model_dir / "norms.pt", map_location=device),
        "concepts": (model_dir / "concepts.txt").read_text().split("\n"),
        "n_concepts": cavs.shape[0],
    }
    for k in ("proj_mean", "proj_std"):
        p = model_dir / f"{k}.pt"
        if p.exists():
            art[k] = torch.load(p, map_location=device)
    if (model_dir / "W_g.pt").exists():
        art["W_g"] = torch.load(model_dir / "W_g.pt", map_location=device)
        art["b_g"] = torch.load(model_dir / "b_g.pt", map_location=device)
    return art


# --- deployed forward path ---------------------------------------------------

def raw_acts_from_emb(z: torch.Tensor, art: dict) -> torch.Tensor:
    """z [N,512] → raw PCBM concept projection (pre-standardisation).

    PCBM paper (Yuksekgonul et al., ICLR 2023, §2):
        f_C^(i)(x) = ⟨f(x), c_i⟩ / ‖c_i‖₂²        (no intercept, ‖·‖ squared)
    the orthogonal projection coefficient of the CLIP feature onto concept c_i.
    This is the space `d_k` is measured in, and the same projC operator DCBM
    uses — so all three CBMs share one concept-projection definition.

    NB — paper vs. official code: the released `models/pcbm_utils.py::compute_dist`
    instead computes the signed SVM margin (⟨f,c⟩+b)/‖c‖ (first power + intercept).
    We follow the *paper*. The two differ only by a per-concept affine map
    (factor 1/‖c_k‖, offset −b_k/‖c_k‖²), which the downstream per-concept
    standardization removes exactly — so the deployed head/accuracy is identical
    either way; only the raw `d_k` coordinate changes. For the mm bank ‖c‖=1, so
    the two forms coincide (mm `d_k` is unaffected by the switch).
    """
    return (z.float() @ art["cavs"].t()) / (art["norms"] ** 2)


def std_acts_from_emb(z: torch.Tensor, art: dict) -> torch.Tensor:
    """z [N,512] → standardized concept activations (what the head reads)."""
    return (raw_acts_from_emb(z, art) - art["proj_mean"]) / art["proj_std"]


def logits_from_std(a: torch.Tensor, art: dict, keep_idx=None) -> torch.Tensor:
    W, b = art["W_g"], art["b_g"]
    if keep_idx is not None:
        return a[:, keep_idx] @ W[:, keep_idx].t() + b
    return a @ W.t() + b


# --- CLIP text encoding (for the multimodal bank) ----------------------------

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
