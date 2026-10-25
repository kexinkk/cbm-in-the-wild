"""Download + selectively extract ImageNet-C for the 4 selected corruption types.

Hendrycks ImageNet-C is distributed as 4 category tars on Zenodo (record 2235448):
    blur.tar     — defocus_blur, glass_blur, motion_blur, zoom_blur
    noise.tar    — gaussian_noise, impulse_noise, shot_noise
    digital.tar  — contrast, elastic_transform, jpeg_compression, pixelate
    weather.tar  — brightness, fog, frost, snow

We evaluate only motion_blur, gaussian_noise,
jpeg_compression, fog. We download the 4 parent tars, extract the 4 chosen
subdirs only, and (by default) delete the tars to recover ~75 GB of disk.

Output layout (all 4 extracted subtrees side-by-side under one root):
    data/raw/ImageNet-C/motion_blur/{1..5}/<wnid>/*.JPEG
    data/raw/ImageNet-C/gaussian_noise/{1..5}/<wnid>/*.JPEG
    data/raw/ImageNet-C/jpeg_compression/{1..5}/<wnid>/*.JPEG
    data/raw/ImageNet-C/fog/{1..5}/<wnid>/*.JPEG

Skip-if-exists: presence of `<corruption>/5/` with at least one wnid subdir
signals prior successful extraction for that corruption.

Usage (~1 h wall-clock):
    python scripts/data_prep/imagenet_c_download.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

from cbm_in_the_wild.paths import imagenet_c_dir

ZENODO_RECORD = "2235448"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# Map each selected corruption to the parent category tar that contains it.
SELECTED = {
    "motion_blur": "blur.tar",
    "gaussian_noise": "noise.tar",
    "jpeg_compression": "digital.tar",
    "fog": "weather.tar",
}


def already_extracted(target_root: Path, corruption: str) -> bool:
    sev5 = target_root / corruption / "5"
    if not sev5.is_dir():
        return False
    return any(p.is_dir() for p in sev5.iterdir())


def download_tar(tar_name: str, dest_dir: Path) -> Path:
    dest = dest_dir / tar_name
    if dest.exists() and dest.stat().st_size > 0:
        size_gb = dest.stat().st_size / 1e9
        print(f"[download] {tar_name} already present ({size_gb:.1f} GB); skipping")
        return dest
    url = f"{ZENODO_URL}/{tar_name}"
    print(f"[download] {url} → {dest}", flush=True)
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-o", str(dest), url],
        check=True,
    )
    return dest


def _peek_tar_layout(tar_path: Path, sample: int = 6) -> list[str]:
    with tarfile.open(tar_path, "r") as tf:
        names: list[str] = []
        for member in tf:
            names.append(member.name)
            if len(names) >= sample:
                break
        return names


def extract_corruption(tar_path: Path, corruption: str, target_root: Path) -> int:
    target_root.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {corruption} from {tar_path.name} → {target_root}", flush=True)
    sample = _peek_tar_layout(tar_path)
    print(f"[extract]   first members: {sample}", flush=True)

    n = 0
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            parts = member.name.split("/")
            if corruption not in parts:
                continue
            idx = parts.index(corruption)
            rewritten = "/".join(parts[idx:])
            member.name = rewritten
            tf.extract(member, target_root)
            n += 1
    if n == 0:
        raise RuntimeError(
            f"{corruption}/ not found in {tar_path.name}; "
            f"first members were {sample}"
        )
    print(f"[extract] {corruption}: {n} entries extracted")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Defaults to imagenet_c_dir() (data/raw/ImageNet-C/).",
    )
    parser.add_argument(
        "--tar-cache",
        type=Path,
        default=None,
        help="Where to stage the downloaded tars (default: <target-root>/_tars/).",
    )
    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="Retain parent tars after selective extract (default: remove to recover disk).",
    )
    args = parser.parse_args()

    target_root = args.target_root or imagenet_c_dir()
    tar_cache = args.tar_cache or (target_root / "_tars")
    target_root.mkdir(parents=True, exist_ok=True)
    tar_cache.mkdir(parents=True, exist_ok=True)

    print(f"[config] target_root = {target_root}")
    print(f"[config] tar_cache   = {tar_cache}")
    print(f"[config] keep_tars   = {args.keep_tars}")

    for corruption, tar_name in SELECTED.items():
        if already_extracted(target_root, corruption):
            print(f"[skip] {corruption}: already extracted under {target_root}")
            continue
        tar_path = download_tar(tar_name, tar_cache)
        extract_corruption(tar_path, corruption, target_root)
        if not args.keep_tars:
            print(f"[cleanup] removing {tar_path}")
            tar_path.unlink()

    if not args.keep_tars and tar_cache.exists() and not any(tar_cache.iterdir()):
        tar_cache.rmdir()

    print("[done] ImageNet-C selective download + extract complete.")


if __name__ == "__main__":
    sys.exit(main())
