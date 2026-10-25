"""LF-CBM training on cached CLIP ViT-B/16 embeddings.

Stage A (one 25-min GPU job): concept text features → clip-cutoff filter →
W_c projection (upstream cos³ objective, val early-stop) → interpretability
filter → train mean/std. Mirrors upstream train_cbm.py step-for-step with
target_features == CLIP image embeddings (backbone clip_ViT-B/16), except
that the concept-similarity matrix is formed per-batch on GPU instead of
materialized (identical values).

Stage B (chained 25-min GPU jobs): sparse GLM-SAGA head via upstream
glm_saga with native state resume (a_table + grad averages persisted to
the workspace between chunks). Warm restarts are exact, not approximate.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from cbm_in_the_wild.paths import DATA_DIR
from cbm_in_the_wild.seeding import set_all_seeds

from .core import (
    LFCBM_REPO,
    add_lfcbm_to_path,
    encode_concept_texts,
    load_artifacts,
    load_emb_split,
    save_head,
    save_stage_a,
)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True)


def run_stage_a(cfg: dict, exp_dir: Path) -> None:
    add_lfcbm_to_path()
    from similarity import cos_similarity_cubed_single

    set_all_seeds(int(cfg["seed"]))
    device = "cuda"
    model_dir = exp_dir / cfg.get("model_subdir", "model")   # per-seed dirs (2026-10-07)
    if (model_dir / "stage_a_meta.json").exists():
        print("[stage-a] already done, skipping")
        return

    dataset = cfg["dataset"]
    concept_file = LFCBM_REPO / cfg["concept_set"]
    concepts = [c for c in concept_file.read_text().split("\n") if c.strip()]
    print(f"[stage-a] {dataset}: {len(concepts)} raw concepts")

    _, Xtr, _, _ = load_emb_split(dataset, "train")
    _, Xva, _, _ = load_emb_split(dataset, "val")
    print(f"[stage-a] train {tuple(Xtr.shape)}, val {tuple(Xva.shape)}")

    text = encode_concept_texts(concepts, cfg["clip_name"], device).float()
    text = _norm(text).to(device)

    # target features for a clip_* backbone == the CLIP image embeddings
    # themselves; the concept-similarity matrix uses the L2-normalized copy.
    img_n = _norm(Xtr)
    val_img_n = _norm(Xva)

    # clip-cutoff filter: mean of top-5 per-concept CLIP activation (streamed)
    topk = torch.full((5, len(concepts)), -1e9, device=device)
    for i in range(0, len(img_n), 50_000):
        sims = img_n[i : i + 50_000].to(device) @ text.t()
        topk = torch.topk(torch.cat([topk, sims]), k=5, dim=0).values
    highest = topk.mean(0).cpu()
    cut = float(cfg["clip_cutoff"])
    keep = highest > cut
    concepts = [c for c, k in zip(concepts, keep) if k]
    text = text[keep.to(device)]
    print(f"[stage-a] clip_cutoff {cut}: {len(concepts)} concepts survive")

    # projection layer, upstream objective + early-stop schedule
    proj = torch.nn.Linear(Xtr.shape[1], len(concepts), bias=False).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    proj_bs = min(int(cfg["proj_batch_size"]), len(Xtr))
    val_clip = (val_img_n.to(device) @ text.t())  # [Nval, K] fits GPU
    val_target = Xva.to(device)

    indices = list(range(len(Xtr)))
    best_val_loss, best_step, best_w = float("inf"), 0, None
    for step in range(int(cfg["proj_steps"])):
        batch = torch.LongTensor(random.sample(indices, k=proj_bs))
        outs = proj(Xtr[batch].to(device))
        clip_batch = img_n[batch].to(device) @ text.t()
        loss = -torch.mean(cos_similarity_cubed_single(clip_batch.detach(), outs))
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == int(cfg["proj_steps"]) - 1:
            with torch.no_grad():
                val_loss = -torch.mean(
                    cos_similarity_cubed_single(val_clip, proj(val_target))
                ).item()
            if step == 0 or val_loss < best_val_loss:
                best_val_loss, best_step = val_loss, step
                best_w = proj.weight.detach().clone()
                print(f"[proj] step {step} val_sim {-val_loss:.4f}")
            else:
                print(f"[proj] early stop at step {step}")
                break
        opt.zero_grad()
    proj.load_state_dict({"weight": best_w})

    # interpretability filter on val
    with torch.no_grad():
        sim = cos_similarity_cubed_single(val_clip, proj(val_target)).cpu()
    icut = float(cfg["interpretability_cutoff"])
    interp = sim > icut
    concepts = [c for c, k in zip(concepts, interp) if k]
    W_c = proj.weight.detach().cpu()[interp]
    print(f"[stage-a] interpretability {icut}: {len(concepts)} concepts final")

    with torch.no_grad():
        Wg = W_c.to(device)
        n, s1, s2 = 0, 0.0, 0.0
        for i in range(0, len(Xtr), 100_000):
            c = Xtr[i : i + 100_000].to(device) @ Wg.t()
            n += c.shape[0]
            s1 = s1 + c.sum(0, dtype=torch.float64)
            s2 = s2 + (c.double() ** 2).sum(0)
        proj_mean = (s1 / n).float().unsqueeze(0).cpu()
        proj_std = ((s2 / (n - 1) - (s1 / n) ** 2 * n / (n - 1))
                    .clamp_min(1e-12).sqrt().float().unsqueeze(0).cpu())

    save_stage_a(
        model_dir, W_c, proj_mean, proj_std, concepts,
        meta={
            "dataset": dataset,
            "n_concepts_raw": int(keep.numel()),
            "n_concepts": len(concepts),
            "best_proj_step": best_step,
            "best_val_similarity": -best_val_loss,
            "clip_cutoff": cut,
            "interpretability_cutoff": icut,
            "seed": int(cfg["seed"]),
        },
    )
    print("[stage-a] saved", model_dir)


def run_stage_b_chunk(cfg: dict, exp_dir: Path) -> bool:
    """One ≤25-min GLM-SAGA chunk. Returns True when the head is finished."""
    add_lfcbm_to_path()
    from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

    set_all_seeds(int(cfg["seed"]))
    device = "cuda"
    model_dir = exp_dir / cfg.get("model_subdir", "model")
    if (model_dir / "head_meta.json").exists():
        print("[stage-b] head already done")
        return True

    dataset = cfg["dataset"]
    saga_cfg = cfg["saga"]
    state_dir = DATA_DIR / "lfcbm" / dataset
    state_dir.mkdir(parents=True, exist_ok=True)
    sub = cfg.get("model_subdir", "model")   # one SAGA state per model dir, so seeds can run in parallel
    state_path = state_dir / ("saga_state.pt" if sub == "model" else f"saga_state_{sub}.pt")
    prog_path = model_dir / "saga_progress.json"

    art = load_artifacts(model_dir, device=device)
    _, Xtr, ytr, classes = load_emb_split(dataset, "train")
    _, Xva, yva, _ = load_emb_split(dataset, "val")

    def _project(X):  # GPU-chunked, result on CPU fp32
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 100_000):
                c = X[i : i + 100_000].to(device) @ art["W_c"].t()
                outs.append(((c - art["proj_mean"]) / art["proj_std"]).cpu())
        return torch.cat(outs)

    train_c = _project(Xtr)
    val_c = _project(Xva)
    del Xtr, Xva

    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    epochs_done, state = 0, None
    if prog_path.exists() and state_path.exists():
        prog = json.loads(prog_path.read_text())
        epochs_done = prog["epochs_done"]
        blob = torch.load(state_path, map_location="cpu")
        state = {
            "a_table": blob["a_table"],
            "w_grad_avg": blob["w_grad_avg"].to(device),
            "b_grad_avg": blob["b_grad_avg"].to(device),
        }
        linear.load_state_dict(blob["linear"])
        print(f"[stage-b] resuming at epoch {epochs_done}")

    n_total = int(saga_cfg["n_iters"])
    loader = DataLoader(
        IndexedTensorDataset(train_c, ytr),
        batch_size=int(saga_cfg["batch_size"]), shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_c, yva),
        batch_size=int(saga_cfg["batch_size"]), shuffle=False,
    )

    def _save_state(st):
        tmp = state_path.with_suffix(".tmp")
        torch.save(
            {
                "a_table": st["a_table"].cpu(),
                "w_grad_avg": st["w_grad_avg"].cpu(),
                "b_grad_avg": st["b_grad_avg"].cpu(),
                "linear": {k: v.cpu() for k, v in linear.state_dict().items()},
            },
            tmp,
        )
        tmp.rename(state_path)

    # Self-pacing inner loop: glm_saga's wall behavior is unknown a priori
    # (epoch time varies with K and node), and a wall kill mid-call loses the
    # chunk. So run 1..chunk_epochs epochs per call, persist state after each
    # call, and size the next call from the measured per-epoch time.
    budget = int(os.environ.get("TIME_BUDGET_S", 1290))
    t_job = time.time()
    per_epoch = None
    params = None
    while epochs_done < n_total:
        elapsed = time.time() - t_job
        if per_epoch is None:
            n = 1
        else:
            fit = int((budget - elapsed - 150) / (per_epoch * 1.25))
            n = max(0, min(int(saga_cfg["chunk_epochs"]), n_total - epochs_done, fit))
        if n == 0:
            print(f"[stage-b] wall budget reached at {epochs_done}/{n_total}")
            break
        t0 = time.time()
        out = glm_saga(
            linear, loader, 0.1, n, 0.99, epsilon=1, k=1,
            val_loader=val_loader, do_zero=False, verbose=1,
            metadata={"max_reg": {"nongrouped": float(saga_cfg["lam"])}},
            n_ex=len(train_c), n_classes=len(classes), state=state,
        )
        dur = time.time() - t0
        per_epoch = dur / n  # includes the per-call loss-eval overhead
        state = out["state"]
        params = out["path"][0]
        epochs_done += n
        _save_state(state)
        prog_path.write_text(json.dumps({
            "epochs_done": epochs_done,
            "val_acc_so_far": params["metrics"]["acc_val"],
        }))
        print(f"[stage-b] +{n} epochs in {dur:.0f}s (~{per_epoch:.0f}s/ep), "
              f"total {epochs_done}/{n_total}, "
              f"val_acc {params['metrics']['acc_val']:.4f}", flush=True)

    if params is not None and epochs_done >= n_total:
        W_g, b_g = params["weight"], params["bias"]
        nnz = int((W_g.abs() > 1e-5).sum())
        save_head(model_dir, W_g, b_g, meta={
            "lam": float(saga_cfg["lam"]),
            "epochs": epochs_done,
            "metrics": params["metrics"],
            "nnz": nnz,
            "sparsity": nnz / W_g.numel(),
        })
        state_path.unlink(missing_ok=True)
        print(f"[stage-b] DONE val_acc={params['metrics']['acc_val']:.4f} "
              f"sparsity={nnz}/{W_g.numel()}")
        return True

    return False
