"""Compute PSNR / SSIM baselines vs clean for 10 corruption types × 5 severities.

Outputs per-condition mean ± std over a class-stratified subset. Does not
involve DCBM / CLIP; purely an image-similarity calibration step.

Usage:
    python experiments/corruption_baselines/run.py --dataset cifar10
    python experiments/corruption_baselines/run.py --dataset imagenet \\
        --subset 500 --seed 42

Output: writes / updates `experiments/corruption_baselines/summary.json`
with a top-level key per dataset (merged with any existing run).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import datasets as tv_datasets
from torchvision import transforms as T

from cbm_in_the_wild.corruptions import (
    CIFAR10C_SELECTED,
    IMAGENET_C_SELECTED,
    SENSOR_TRANSFORMS,
    imagenet_c_severity_dir,
    load_cifar10c,
)
from cbm_in_the_wild.paths import DATA_DIR, imagenet_dir

SEVERITIES = (1, 2, 3, 4, 5)
HERE = Path(__file__).resolve().parent


# ---------- subset selection ----------

def stratified_indices(labels: np.ndarray, n_per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Return stratified sample indices: n_per_class per unique label."""
    out: list[int] = []
    for c in np.unique(labels):
        class_idx = np.flatnonzero(labels == c)
        pick = rng.choice(class_idx, size=min(n_per_class, len(class_idx)), replace=False)
        out.extend(pick.tolist())
    return np.sort(np.array(out, dtype=np.int64))


# ---------- metrics ----------

def psnr_ssim_per_image(clean_hwc: np.ndarray, corrupted_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """clean/corrupted: uint8 arrays of shape (N, H, W, 3). Returns PSNR, SSIM arrays len N."""
    n = len(clean_hwc)
    psnr = np.empty(n, dtype=np.float64)
    ssim = np.empty(n, dtype=np.float64)
    for i in range(n):
        psnr[i] = peak_signal_noise_ratio(clean_hwc[i], corrupted_hwc[i], data_range=255)
        ssim[i] = structural_similarity(
            clean_hwc[i], corrupted_hwc[i], data_range=255, channel_axis=2
        )
    return psnr, ssim


def aggregate(psnr: np.ndarray, ssim: np.ndarray) -> dict:
    # PSNR may be +inf when clean == corrupted; skip inf entries in the mean.
    finite = np.isfinite(psnr)
    return {
        "psnr_mean": float(psnr[finite].mean()) if finite.any() else float("inf"),
        "psnr_std": float(psnr[finite].std()) if finite.any() else 0.0,
        "ssim_mean": float(ssim.mean()),
        "ssim_std": float(ssim.std()),
        "n": int(len(psnr)),
        "n_inf_psnr": int((~finite).sum()),
    }


# ---------- sensor-transform application ----------

def apply_sensor_uint8(clean_hwc: np.ndarray, name: str, severity: int, base_seed: int) -> np.ndarray:
    """Apply a sensor transform to uint8 HWC images; return uint8 HWC."""
    cls = SENSOR_TRANSFORMS[name]
    out = np.empty_like(clean_hwc)
    for i in range(len(clean_hwc)):
        tensor_chw = torch.from_numpy(clean_hwc[i]).permute(2, 0, 1)  # uint8 CHW
        g = torch.Generator().manual_seed(base_seed * 100_000 + severity * 1_000 + i)
        corrupted = cls(severity=severity, generator=g)(tensor_chw)
        out[i] = corrupted.permute(1, 2, 0).numpy().astype(np.uint8)
    return out


# ---------- CIFAR-10 pipeline ----------

def run_cifar10(subset_size: int, seed: int) -> dict:
    clean_ds = tv_datasets.CIFAR10(str(DATA_DIR / "raw"), train=False, download=False)
    labels = np.asarray(clean_ds.targets)
    rng = np.random.default_rng(seed)
    indices = stratified_indices(labels, n_per_class=subset_size // 10, rng=rng)
    clean_subset = np.asarray(clean_ds.data)[indices]  # uint8 HWC (N, 32, 32, 3)

    results: dict = {}
    # 4 Hendrycks natural corruptions.
    for corruption in CIFAR10C_SELECTED:
        results[corruption] = {}
        for severity in SEVERITIES:
            corr_full, _ = load_cifar10c(corruption, severity)
            corr_subset = np.asarray(corr_full[indices])
            psnr, ssim = psnr_ssim_per_image(clean_subset, corr_subset)
            results[corruption][f"severity_{severity}"] = aggregate(psnr, ssim)
            print(f"  cifar10/{corruption:20s} sev{severity}: "
                  f"psnr={psnr.mean():.2f}±{psnr.std():.2f}, ssim={ssim.mean():.3f}±{ssim.std():.3f}",
                  flush=True)
    # 6 custom sensor-level corruptions.
    for name in SENSOR_TRANSFORMS:
        results[name] = {}
        for severity in SEVERITIES:
            corr_subset = apply_sensor_uint8(clean_subset, name, severity, seed)
            psnr, ssim = psnr_ssim_per_image(clean_subset, corr_subset)
            results[name][f"severity_{severity}"] = aggregate(psnr, ssim)
            print(f"  cifar10/{name:25s} sev{severity}: "
                  f"psnr={psnr.mean():.2f}±{psnr.std():.2f}, ssim={ssim.mean():.3f}±{ssim.std():.3f}",
                  flush=True)
    return {"subset_indices": indices.tolist()[:5] + ["..."], "n_subset": len(indices), "corruptions": results}


# ---------- ImageNet pipeline ----------

def _imagenet_eval_preproc(size: int = 224, resize: int = 256):
    return T.Compose([T.Resize(resize), T.CenterCrop(size), T.PILToTensor()])


def _load_imagenet_clean_subset(
    n_classes_sampled: int, imgs_per_class: int, seed: int, size: int = 224, resize: int = 256
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Return uint8 HWC clean subset of shape (N, size, size, 3) and list of (wnid, filename)."""
    # ImageNet-C is published over the ILSVRC2012 val set, mounted at
    # data/imagenet/test/ here. data/imagenet/val/ is the train-carved
    # 10% (training-style filenames) and won't align to ImageNet-C wnid/file.
    test_root = imagenet_dir() / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"clean ImageNet test (= ILSVRC val) not found at {test_root}")
    wnids = sorted(p.name for p in test_root.iterdir() if p.is_dir())
    rng = np.random.default_rng(seed)
    chosen_wnids = rng.choice(wnids, size=min(n_classes_sampled, len(wnids)), replace=False)
    preproc = _imagenet_eval_preproc(size=size, resize=resize)

    arrs: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    for wnid in chosen_wnids:
        wnid_dir = test_root / wnid
        files = sorted(p.name for p in wnid_dir.iterdir() if p.is_file())
        pick = rng.choice(files, size=min(imgs_per_class, len(files)), replace=False)
        for fname in pick:
            with Image.open(wnid_dir / fname) as im:
                im = im.convert("RGB")
                tensor_chw = preproc(im)  # uint8 CHW
            arrs.append(tensor_chw.permute(1, 2, 0).numpy())
            keys.append((str(wnid), str(fname)))
    return np.stack(arrs, axis=0), keys


def _load_imagenet_c_matching(
    corruption: str,
    severity: int,
    keys: list[tuple[str, str]],
    size: int = 224,
) -> np.ndarray:
    """Load ImageNet-C images matching the given (wnid, filename) keys."""
    sev_dir = imagenet_c_severity_dir(corruption, severity)
    preproc = T.Compose([T.CenterCrop(size), T.PILToTensor()])
    out = np.empty((len(keys), size, size, 3), dtype=np.uint8)
    for i, (wnid, fname) in enumerate(keys):
        with Image.open(sev_dir / wnid / fname) as im:
            im = im.convert("RGB")
            out[i] = preproc(im).permute(1, 2, 0).numpy()
    return out


def run_imagenet(subset_size: int, seed: int) -> dict:
    clean_subset, keys = _load_imagenet_clean_subset(
        n_classes_sampled=subset_size,
        imgs_per_class=1,
        seed=seed,
    )
    print(f"  imagenet: clean subset loaded, shape {clean_subset.shape}", flush=True)

    results: dict = {}
    for corruption in IMAGENET_C_SELECTED:
        results[corruption] = {}
        for severity in SEVERITIES:
            corr_subset = _load_imagenet_c_matching(corruption, severity, keys)
            psnr, ssim = psnr_ssim_per_image(clean_subset, corr_subset)
            results[corruption][f"severity_{severity}"] = aggregate(psnr, ssim)
            print(f"  imagenet/{corruption:20s} sev{severity}: "
                  f"psnr={psnr.mean():.2f}±{psnr.std():.2f}, ssim={ssim.mean():.3f}±{ssim.std():.3f}",
                  flush=True)
    for name in SENSOR_TRANSFORMS:
        results[name] = {}
        for severity in SEVERITIES:
            corr_subset = apply_sensor_uint8(clean_subset, name, severity, seed)
            psnr, ssim = psnr_ssim_per_image(clean_subset, corr_subset)
            results[name][f"severity_{severity}"] = aggregate(psnr, ssim)
            print(f"  imagenet/{name:25s} sev{severity}: "
                  f"psnr={psnr.mean():.2f}±{psnr.std():.2f}, ssim={ssim.mean():.3f}±{ssim.std():.3f}",
                  flush=True)
    return {"n_subset": len(clean_subset), "corruptions": results}


# ---------- entry ----------

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--subset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()

    print(f"[config] dataset={args.dataset}, subset={args.subset}, seed={args.seed}", flush=True)
    t0 = time.time()
    if args.dataset == "cifar10":
        dataset_results = run_cifar10(args.subset, args.seed)
    else:
        dataset_results = run_imagenet(args.subset, args.seed)
    elapsed = time.time() - t0

    # Merge with existing summary (preserve the other dataset's results if present).
    if args.out.exists():
        with open(args.out) as f:
            summary = json.load(f)
    else:
        summary = {"config": {"severities": list(SEVERITIES)}, "datasets": {}}
    summary.setdefault("datasets", {})
    summary["datasets"][args.dataset] = {
        "config": {"subset": args.subset, "seed": args.seed},
        "git_commit": git_sha(),
        "elapsed_seconds": round(elapsed, 1),
        **dataset_results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {args.out} (elapsed {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
