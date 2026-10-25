"""LF-CBM: stage-wise corruption propagation on cached embeddings.

Emits the same row schema as the DCBM propagation experiments
(clean_baseline / clip_embedding / concept_activation / prediction), with
concept-stage rank metrics computed on the *standardized* activations —
the space the sparse head actually reads (mirrors run_scaled_metrics.py).
LF-CBM has no concept softmax, so entropy/KL rows are not emitted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from cbm_in_the_wild.metrics import (
    argmax_shift_fraction,
    spearman_rho,
    top_k_concept_overlap,
)

from .cells import load_cell
from .core import load_artifacts, load_emb_split, logits_from_std, std_acts_from_emb


def _clean_reference(cfg: dict, exp_dir: Path, device: str):
    ids, X, y, _ = load_emb_split(cfg["dataset"], "test")
    art = load_artifacts(exp_dir / "model", device=device)
    with torch.no_grad():
        acts = std_acts_from_emb(X.to(device), art)
        preds = logits_from_std(acts, art).argmax(1)
    row_of = {i: k for k, i in enumerate(ids)}
    return {
        "row_of": row_of, "emb": X, "acts": acts, "preds": preds,
        "labels": y.to(device), "art": art,
    }


def run_cells(cfg: dict, exp_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = _clean_reference(cfg, exp_dir, device)
    out_path = exp_dir / "stage_metrics.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if r["stage"] == "prediction":
                done.add((r["corruption"], r["severity"]))

    clean_acc = float((ref["preds"] == ref["labels"]).float().mean())
    rows = []
    if not done:
        rows.append({
            "stage": "clean_baseline", "dataset": cfg["dataset"],
            "n": len(ref["labels"]), "clean_eval_accuracy": clean_acc,
        })
    print(f"[prop] clean acc {clean_acc:.4f}; {len(done)} cells already done")

    from cbm_in_the_wild.corruption_eval import resolve_corruptions

    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        for corruption in resolve_corruptions(cfg["corruption_set"]):
            for severity in cfg["severities"]:
                if (corruption, severity) in done:
                    continue
                t0 = time.time()
                cell = load_cell(cfg, exp_dir, corruption, severity)
                idx = torch.tensor(
                    [ref["row_of"][i] for i in cell["image_ids"]], dtype=torch.long
                )
                z_cl = ref["emb"][idx].to(device)
                z_co = cell["clip_emb"].to(device)
                a_cl = ref["acts"][idx.to(device)]
                labels = ref["labels"][idx.to(device)]
                preds_cl = ref["preds"][idx.to(device)]

                with torch.no_grad():
                    cos = torch.nn.functional.cosine_similarity(z_cl, z_co, dim=1)
                    a_co = std_acts_from_emb(z_co, ref["art"])
                    preds_co = logits_from_std(a_co, ref["art"]).argmax(1)

                    sp = spearman_rho(a_cl, a_co)
                    tk = top_k_concept_overlap(a_cl, a_co, k=5)
                    asf = argmax_shift_fraction(a_cl, a_co)
                    pcc = torch.nn.functional.cosine_similarity(
                        a_cl - a_cl.mean(0), a_co - a_co.mean(0), dim=0
                    ).mean()

                base = {"dataset": cfg["dataset"], "corruption": corruption,
                        "severity": severity}
                out_rows = [
                    {"stage": "clip_embedding", **base, "n": len(idx),
                     "cosine_drift_mean": float(cos.mean()),
                     "cosine_drift_std": float(cos.std())},
                    {"stage": "concept_activation", **base,
                     "per_concept_cosine_mean": float(pcc),
                     "top5_jaccard_mean": float(tk.mean()),
                     "argmax_shift_fraction": float(asf),
                     "spearman_rho_mean": float(sp.mean())},
                    {"stage": "prediction", **base,
                     "accuracy": float((preds_co == labels).float().mean()),
                     "prediction_shifts_vs_clean": int((preds_co != preds_cl).sum())},
                ]
                for r in out_rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[prop] {corruption}_s{severity} "
                      f"acc {out_rows[-1]['accuracy']:.4f} ({time.time()-t0:.0f}s)")
    print("[prop] all cells done →", out_path)
