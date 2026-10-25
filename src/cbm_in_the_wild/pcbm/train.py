"""PCBM Stage A: build concept bank(s) + standardizer + reference head.

One short GPU job produces a complete deployed model per concept source:

  cav  — per-concept linear SVM (sklearn SVC, kernel=linear) on CLIP ViT-B/16
         embeddings of Broden concept images. CAV = svm.coef_, with intercept_
         and ‖coef_‖ stored so the deployed margin is the exact PCBM projection
         (concepts/concept_utils.py::get_cavs in the fork). Faithful primary.
  mm   — CLIP text-encoder embeddings of the same concept names, unit-norm
         (intercepts=0, norms=1). Cheap secondary flavor.

Then: standardize over cached ImageNet train projections, fit an L2 multinomial
logistic reference head on a class-stratified train subsample. The head
regularizer is NOT the variable under test — the capacity-matched ablation
refits its own head per (rank, K, seed) — so we use a fast dense L2 head here
rather than PCBM's elastic-net SGD (documented deviation).

Broden images are staged to $TMPDIR by the sbatch; PCBM_BRODEN_DIR points at
the extracted `broden_concepts/` tree (<concept>/{positives,negatives}/*.png).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cbm_in_the_wild.paths import DATA_DIR, REPO_ROOT
from cbm_in_the_wild.seeding import set_all_seeds

# glm_saga (elastic-net multinomial solver) ships with the LF-CBM repo; it's a
# generic optimizer, not project code — importing it is like importing sklearn.
LFCBM_REPO = Path(os.environ.get("LFCBM_REPO", Path.home() / "Label-free-CBM"))

from .core import (
    encode_concept_texts,
    load_artifacts,
    load_emb_split,
    save_bank,
    save_head,
    save_proj_stats,
)


# --- CLIP image embedding ----------------------------------------------------

def _clip_load(clip_name: str, device: str):
    import clip

    model, preprocess = clip.load(clip_name, device=device)
    return model.eval(), preprocess


@torch.no_grad()
def _embed_images(paths, model, preprocess, device, bs: int = 256) -> np.ndarray:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    out = []
    for i in range(0, len(paths), bs):
        batch = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths[i : i + bs]]
        z = model.encode_image(torch.cat(batch).to(device))
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


# --- CAV bank ----------------------------------------------------------------

def _valid_concepts(broden_dir: Path) -> list[str]:
    """Concepts with >=1 image in BOTH positives and negatives.

    Deterministic (sorted) and a pure function of the dir, so the cav and mm
    banks — which both call this — stay row-aligned. Broden ships a handful of
    empty concept dirs (e.g. `blurriness/positives` is empty); those are dropped
    here, giving the real K_full.
    """
    out = []
    for d in sorted(broden_dir.iterdir()):
        pos, neg = d / "positives", d / "negatives"
        if (d.is_dir() and pos.is_dir() and neg.is_dir()
                and any(pos.glob("*.png")) and any(neg.glob("*.png"))):
            out.append(d.name)
    return out


def _build_cav_bank(broden_dir: Path, model, preprocess, cfg, device):
    """Mirror the fork's get_cavs: linear SVM per concept, CAV = coef_."""
    from sklearn.svm import SVC

    n = int(cfg["n_samples"])
    C = float(cfg["svm_C"])
    rng = np.random.RandomState(int(cfg["seed"]))
    concepts = _valid_concepts(broden_dir)
    print(f"[cav] {len(concepts)} valid concepts under {broden_dir}")

    def _pick(concept, which):
        files = sorted(str(p) for p in (broden_dir / concept / which).glob("*.png"))
        replace = len(files) < 2 * n
        if replace:
            print(f"[cav]   {concept}/{which}: {len(files)} < {2*n}, sampling w/ replacement")
        return list(rng.choice(files, 2 * n, replace=replace))

    cavs, intercepts, norms, kept, meta = [], [], [], [], []
    for ci, concept in enumerate(concepts):
        pos = _embed_images(_pick(concept, "positives"), model, preprocess, device)
        neg = _embed_images(_pick(concept, "negatives"), model, preprocess, device)
        X_tr = np.concatenate([pos[:n], neg[:n]], axis=0)
        X_va = np.concatenate([pos[n:], neg[n:]], axis=0)
        y_tr = np.concatenate([np.ones(n), np.zeros(n)])
        y_va = np.concatenate([np.ones(len(pos) - n), np.zeros(len(neg) - n)])
        svm = SVC(C=C, kernel="linear")
        svm.fit(X_tr, y_tr)
        cavs.append(svm.coef_.astype(np.float32))          # [1,512]
        intercepts.append(float(svm.intercept_[0]))
        norms.append(float(np.linalg.norm(svm.coef_)))
        kept.append(concept)
        meta.append({"concept": concept, "train_acc": float(svm.score(X_tr, y_tr)),
                     "val_acc": float(svm.score(X_va, y_va))})
        if ci % 20 == 0 or ci == len(concepts) - 1:
            print(f"[cav] {ci+1}/{len(concepts)} {concept} "
                  f"val_acc={meta[-1]['val_acc']:.3f}", flush=True)

    return (torch.tensor(np.concatenate(cavs, axis=0)),
            torch.tensor(intercepts, dtype=torch.float32),
            torch.tensor(norms, dtype=torch.float32),
            kept, meta)


def _build_mm_bank(concepts, cfg, device):
    """Multimodal bank: unit-norm CLIP text features; intercepts=0, norms=1."""
    prompts = [c.replace("_", " ") for c in concepts]
    text = encode_concept_texts(prompts, cfg["clip_name"], device).float()
    text = text / text.norm(dim=1, keepdim=True)
    K = len(concepts)
    return (text, torch.zeros(K), torch.ones(K), list(concepts),
            [{"concept": c, "prompt": p} for c, p in zip(concepts, prompts)])


# --- standardizer + reference head ------------------------------------------

def _proj_stats(Xtr, art, device):
    from .core import raw_acts_from_emb

    n, s1, s2 = 0, 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(Xtr), 100_000):
            c = raw_acts_from_emb(Xtr[i : i + 100_000].to(device), art)
            n += c.shape[0]
            s1 = s1 + c.sum(0, dtype=torch.float64)
            s2 = s2 + (c.double() ** 2).sum(0)
    mean = (s1 / n).float().unsqueeze(0).cpu()
    std = ((s2 / (n - 1) - (s1 / n) ** 2 * n / (n - 1))
           .clamp_min(1e-12).sqrt().float().unsqueeze(0).cpu())
    return mean, std


def _stratified_idx(y, n_total, seed):
    g = torch.Generator().manual_seed(seed)
    per = max(1, n_total // len(y.unique()))
    picks = [pool[torch.randperm(len(pool), generator=g)[:per]]
             for c in y.unique()
             for pool in [(y == c).nonzero(as_tuple=True)[0]]]
    return torch.cat(picks)


def _fit_logistic(Xf, yf, n_classes, device, max_iter=800):
    Xf, yf = Xf.to(device), yf.to(device)
    W = torch.zeros(n_classes, Xf.shape[1], device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xf @ W.t() + b, yf, reduction="sum") + 0.5 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


@torch.no_grad()
def _eval_acc(X, y, art, W, b, device, bs=100_000):
    from .core import std_acts_from_emb

    hits = 0
    for i in range(0, len(X), bs):
        a = std_acts_from_emb(X[i : i + bs].to(device), art)
        hits += int((a @ W.t() + b).argmax(1).eq(y[i : i + bs].to(device)).sum())
    return hits / len(X)


# --- stage A entrypoint ------------------------------------------------------

def _model_dir(exp_dir: Path, bank: str, cfg: dict) -> Path:
    """model_<bank><model_suffix>; the suffix (e.g. _seed1, 2026-10-11) keeps 3-seed
    replicates apart from the deployed single-seed dirs, which stay untouched."""
    return exp_dir / f"model_{bank}{cfg.get('model_suffix', '')}"


def _build_one(bank: str, cfg, exp_dir, model, preprocess, broden_dir, device):
    model_dir = _model_dir(exp_dir, bank, cfg)
    if (model_dir / "head_meta.json").exists():
        print(f"[stage-a:{bank}] complete, skipping")
        return
    dataset = cfg["dataset"]

    # 1. concept bank
    if bank == "cav":
        cavs, intercepts, norms, concepts, cmeta = _build_cav_bank(
            broden_dir, model, preprocess, cfg, device)
    else:
        # same valid-concept set as cav (row-aligned banks)
        concepts = _valid_concepts(broden_dir)
        cavs, intercepts, norms, concepts, cmeta = _build_mm_bank(concepts, cfg, device)
    save_bank(model_dir, cavs, intercepts, norms, concepts,
              meta={"bank": bank, "n_concepts": len(concepts), "seed": int(cfg["seed"]),
                    "svm_C": cfg.get("svm_C"), "n_samples": cfg.get("n_samples"),
                    "concepts": cmeta})
    art = load_artifacts(model_dir, device=device)
    print(f"[stage-a:{bank}] bank {tuple(art['cavs'].shape)}")

    # 2. standardizer over cached ImageNet train projections
    _, Xtr, ytr, classes = load_emb_split(dataset, "train")
    mean, std = _proj_stats(Xtr, art, device)
    save_proj_stats(model_dir, mean, std)
    art["proj_mean"], art["proj_std"] = mean.to(device), std.to(device)

    # 3. reference head — L2 multinomial logistic on a stratified train subsample
    fit_n = int(cfg.get("head_fit_n", 250_000))
    idx = _stratified_idx(ytr, fit_n, seed=int(cfg["seed"]))
    with torch.no_grad():
        from .core import std_acts_from_emb
        Af = std_acts_from_emb(Xtr[idx].to(device), art).cpu()
    W, b = _fit_logistic(Af, ytr[idx], len(classes), device)

    _, Xte, yte, _ = load_emb_split(dataset, "test")
    test_acc = _eval_acc(Xte, yte, art, W, b, device)
    # NOTE: this is a fast L2 *sanity* head, not the deployed model. Stage B
    # (glm_saga elastic-net) overwrites W_g/b_g with the faithful PCBM head.
    save_head(model_dir, W, b, meta={"head": "l2_sanity_pre_stageB",
                                     "fit_n": len(idx), "n_concepts": len(concepts),
                                     "clean_test_top1_l2_sanity": test_acc})
    print(f"[stage-a:{bank}] bank+std+sanity DONE  L2-sanity top-1 = {test_acc:.4f}  "
          f"(K={len(concepts)}); run stage B for the faithful head", flush=True)


def run_stage_a(cfg: dict, exp_dir: Path) -> None:
    set_all_seeds(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    banks = cfg.get("banks", ["cav", "mm"])
    broden_dir = None
    if set(banks) & {"cav", "mm"}:
        broden_dir = Path(os.environ["PCBM_BRODEN_DIR"])
        if not broden_dir.is_dir():
            raise FileNotFoundError(f"PCBM_BRODEN_DIR not found: {broden_dir}")
    model, preprocess = _clip_load(cfg["clip_name"], device)
    for bank in banks:
        _build_one(bank, cfg, exp_dir, model, preprocess, broden_dir, device)


# --- stage aL: large-vocabulary multimodal bank -----------------------------

def run_stage_a_mm_large(cfg: dict, exp_dir: Path) -> None:
    """Build the large-vocabulary multimodal bank (mmL): CLIP-text embeddings of
    an arbitrary concept list (default: LF-CBM's deployed ~4.6k ImageNet
    concepts), unit-norm — same forward as `mm`, no Broden. Tests whether PCBM's
    smaller pruning margin on the 166-concept banks is a bank-size artifact by
    giving it a DCBM/LF-CBM-width concept set. Writes model_mmL/ + an L2 sanity
    head; stage B (glm_saga) then fits the faithful head.
    """
    set_all_seeds(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = "mmL"
    model_dir = _model_dir(exp_dir, bank, cfg)
    if (model_dir / "head_meta.json").exists():
        print(f"[stage-a:{bank}] complete, skipping")
        return
    dataset = cfg["dataset"]

    cfile = Path(cfg["mm_large_concept_file"])
    if not cfile.is_absolute():
        cfile = REPO_ROOT / cfg["mm_large_concept_file"]
    concepts = [c.strip() for c in cfile.read_text().splitlines() if c.strip()]
    print(f"[stage-a:{bank}] {len(concepts)} concepts from {cfile}")

    cavs, intercepts, norms, concepts, cmeta = _build_mm_bank(concepts, cfg, device)
    save_bank(model_dir, cavs, intercepts, norms, concepts,
              meta={"bank": bank, "n_concepts": len(concepts), "seed": int(cfg["seed"]),
                    "concept_source": str(cfile), "concepts_sample": cmeta[:20]})
    art = load_artifacts(model_dir, device=device)
    print(f"[stage-a:{bank}] bank {tuple(art['cavs'].shape)}")

    _, Xtr, ytr, classes = load_emb_split(dataset, "train")
    mean, std = _proj_stats(Xtr, art, device)
    save_proj_stats(model_dir, mean, std)
    art["proj_mean"], art["proj_std"] = mean.to(device), std.to(device)

    from .core import std_acts_from_emb
    fit_n = int(cfg.get("head_fit_n", 250_000))
    idx = _stratified_idx(ytr, fit_n, seed=int(cfg["seed"]))
    with torch.no_grad():
        Af = std_acts_from_emb(Xtr[idx].to(device), art).cpu()
    W, b = _fit_logistic(Af, ytr[idx], len(classes), device)

    _, Xte, yte, _ = load_emb_split(dataset, "test")
    test_acc = _eval_acc(Xte, yte, art, W, b, device)
    save_head(model_dir, W, b, meta={"head": "l2_sanity_pre_stageB",
                                     "fit_n": len(idx), "n_concepts": len(concepts),
                                     "clean_test_top1_l2_sanity": test_acc})
    print(f"[stage-a:{bank}] DONE  L2-sanity top-1 = {test_acc:.4f}  "
          f"(K={len(concepts)}); run stage B for the faithful head", flush=True)


# --- refresh: recompute standardizer under a changed projection -------------

def run_refresh_proj(cfg: dict, exp_dir: Path) -> None:
    """Recompute proj_mean/proj_std for the existing bank under the current
    `raw_acts_from_emb` projection (e.g. after switching to the paper's
    ⟨f,c⟩/‖c‖² form). Keeps the bank and the head: per-concept standardization
    is scale/shift-invariant, so std-acts — and thus the deployed head — are
    unchanged; only the stored proj-stats must be made consistent with the new
    raw projection so `std_acts_from_emb` reproduces them. cfg["bank"] selects
    cav|mm. Needs only cached ImageNet embeddings (no Broden, no head retrain).
    """
    set_all_seeds(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = cfg["bank"]
    model_dir = _model_dir(exp_dir, bank, cfg)
    art = load_artifacts(model_dir, device=device)
    _, Xtr, _, _ = load_emb_split(cfg["dataset"], "train")
    mean, std = _proj_stats(Xtr, art, device)
    save_proj_stats(model_dir, mean, std)
    print(f"[refresh:{bank}] proj-stats recomputed under the paper projection "
          f"(⟨f,c⟩/‖c‖²); bank + head unchanged")


# --- stage B: faithful PCBM head (sparse elastic-net via glm_saga) -----------

def run_stage_b_chunk(cfg: dict, exp_dir: Path) -> bool:
    """One <=25-min glm_saga chunk for the faithful PCBM reference head.

    Sparse elastic-net (alpha=0.99) multinomial head — PCBM's published head
    type (train_pcbm.py uses SGDClassifier(penalty='elasticnet'); glm_saga is
    the same objective, solved on GPU). Projects ImageNet train through the
    frozen bank + standardizer, then fits. Overwrites the Stage-A L2 sanity
    head. Chunked + resumable (native glm_saga state) like the LF-CBM head;
    at K=166 it typically finishes in one wave. cfg["bank"] selects cav|mm.
    """
    if str(LFCBM_REPO) not in sys.path:
        sys.path.insert(0, str(LFCBM_REPO))
    from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

    from .core import std_acts_from_emb

    set_all_seeds(int(cfg["seed"]))
    device = "cuda"
    bank = cfg["bank"]
    model_dir = _model_dir(exp_dir, bank, cfg)
    head_meta = json.loads((model_dir / "head_meta.json").read_text()) \
        if (model_dir / "head_meta.json").exists() else {}
    if head_meta.get("head") == "elasticnet_glmsaga":
        print(f"[stage-b:{bank}] faithful head already done")
        return True

    dataset = cfg["dataset"]
    saga_cfg = cfg["saga"]
    suffix = cfg.get("model_suffix", "")
    # per-(dataset, bank, seed) resume state; the unsuffixed path is kept for the old single runs
    state_dir = DATA_DIR / "pcbm" / (f"{dataset}_{bank}{suffix}" if suffix else bank)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "saga_state.pt"
    prog_path = model_dir / "saga_progress.json"

    art = load_artifacts(model_dir, device=device)
    _, Xtr, ytr, classes = load_emb_split(dataset, "train")
    _, Xva, yva, _ = load_emb_split(dataset, "val")

    def _project(X):
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 100_000):
                outs.append(std_acts_from_emb(X[i : i + 100_000].to(device), art).cpu())
        return torch.cat(outs)

    train_c, val_c = _project(Xtr), _project(Xva)
    del Xtr, Xva

    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    epochs_done, state = 0, None
    if prog_path.exists() and state_path.exists():
        prog = json.loads(prog_path.read_text())
        epochs_done = prog["epochs_done"]
        blob = torch.load(state_path, map_location="cpu")
        state = {"a_table": blob["a_table"],
                 "w_grad_avg": blob["w_grad_avg"].to(device),
                 "b_grad_avg": blob["b_grad_avg"].to(device)}
        linear.load_state_dict(blob["linear"])
        print(f"[stage-b:{bank}] resuming at epoch {epochs_done}")

    n_total = int(saga_cfg["n_iters"])
    loader = DataLoader(IndexedTensorDataset(train_c, ytr),
                        batch_size=int(saga_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(TensorDataset(val_c, yva),
                            batch_size=int(saga_cfg["batch_size"]), shuffle=False)

    def _save_state(st):
        tmp = state_path.with_suffix(".tmp")
        torch.save({"a_table": st["a_table"].cpu(),
                    "w_grad_avg": st["w_grad_avg"].cpu(),
                    "b_grad_avg": st["b_grad_avg"].cpu(),
                    "linear": {k: v.cpu() for k, v in linear.state_dict().items()}}, tmp)
        tmp.rename(state_path)

    budget = int(os.environ.get("TIME_BUDGET_S", 1290))
    t_job = time.time()
    per_epoch, params = None, None
    while epochs_done < n_total:
        elapsed = time.time() - t_job
        n = 1 if per_epoch is None else max(
            0, min(int(saga_cfg["chunk_epochs"]), n_total - epochs_done,
                   int((budget - elapsed - 150) / (per_epoch * 1.25))))
        if n == 0:
            print(f"[stage-b:{bank}] wall budget reached at {epochs_done}/{n_total}")
            break
        t0 = time.time()
        out = glm_saga(linear, loader, 0.1, n, 0.99, epsilon=1, k=1,
                       val_loader=val_loader, do_zero=False, verbose=1,
                       metadata={"max_reg": {"nongrouped": float(saga_cfg["lam"])}},
                       n_ex=len(train_c), n_classes=len(classes), state=state)
        per_epoch = (time.time() - t0) / n
        state, params = out["state"], out["path"][0]
        epochs_done += n
        _save_state(state)
        prog_path.write_text(json.dumps({"epochs_done": epochs_done,
                                         "val_acc_so_far": params["metrics"]["acc_val"]}))
        print(f"[stage-b:{bank}] +{n} ep (~{per_epoch:.0f}s/ep) "
              f"{epochs_done}/{n_total} val_acc {params['metrics']['acc_val']:.4f}",
              flush=True)

    if params is not None and epochs_done >= n_total:
        W_g, b_g = params["weight"], params["bias"]
        nnz = int((W_g.abs() > 1e-5).sum())
        _, Xte, yte, _ = load_emb_split(dataset, "test")
        test_acc = _eval_acc(Xte, yte, art, W_g.to(device), b_g.to(device), device)
        save_head(model_dir, W_g, b_g, meta={
            "head": "elasticnet_glmsaga", "alpha": 0.99,
            "lam": float(saga_cfg["lam"]), "epochs": epochs_done,
            "metrics": params["metrics"], "nnz": nnz,
            "sparsity": nnz / W_g.numel(), "clean_test_top1": test_acc})
        state_path.unlink(missing_ok=True)
        print(f"[stage-b:{bank}] DONE faithful head clean top-1={test_acc:.4f} "
              f"sparsity={nnz}/{W_g.numel()}")
        return True
    return False
