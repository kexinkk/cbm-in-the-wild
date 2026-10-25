"""Extract ILSVRC2012 tars into DCBM's ImageFolder layout.

Input (read-only):
    data/raw/ILSVRC2012_img_train.tar       (~138 GB, tar-of-tars, 1000 wnid tars)
    data/raw/ILSVRC2012_img_val.tar         (~6.3 GB, 50 000 flat JPEGs)
    data/raw/ILSVRC2012_devkit_t12/data/
        meta.mat                            (wnid <-> int id, ILSVRC2012 order)
        ILSVRC2012_validation_ground_truth.txt (50 000 lines, 1..1000)

Output:
    data/imagenet/train_full/<wnid>/*.JPEG  (1 281 167 files, 1000 classes)
    data/imagenet/test/<wnid>/*.JPEG        (50 000 files, 1000 classes)

train_full/ is the immutable post-extract snapshot; imagenet_split_val.py
carves the 90/10 internal train/val split off it (non-destructive, via
symlinks — 130 GB of data does not get duplicated).

Naming follows Q2 resolution (2026-04-21, dev_log.md): ILSVRC2012 val
IS DCBM's "test" set (their file `images_ImageNet_test_*.torch` is
actually the 50k val — see `create_IN_200.py:12` inline comment). We
adopt DCBM's naming so the upstream `CBM` class finds these files.

Idempotent: re-running skips any wnid dir that already has all its
files extracted. A sentinel file `.extract_done` under each wnid dir
records completion so a partial-extract interruption can be detected.

Usage (activate env first; this is heavy — run via sbatch, not login):
    python scripts/data_prep/imagenet_to_imagefolder.py --dry-run
    python scripts/data_prep/imagenet_to_imagefolder.py
    python scripts/data_prep/imagenet_to_imagefolder.py --split val
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

from cbm_in_the_wild.paths import DATA_DIR, imagenet_dir


RAW_DIR = DATA_DIR / "raw"
TRAIN_TAR = RAW_DIR / "ILSVRC2012_img_train.tar"
VAL_TAR = RAW_DIR / "ILSVRC2012_img_val.tar"
DEVKIT_DATA = RAW_DIR / "ILSVRC2012_devkit_t12" / "data"
META_MAT = DEVKIT_DATA / "meta.mat"
VAL_GT = DEVKIT_DATA / "ILSVRC2012_validation_ground_truth.txt"


def load_wnid_order() -> list[str]:
    """Return wnid list indexed by ILSVRC2012 1-based class id.

    Index 0 is a sentinel empty string so `wnids[class_id]` works with
    1-based labels from the ground-truth file. Result length = 1001.
    """
    import scipy.io

    if not META_MAT.exists():
        raise FileNotFoundError(f"devkit meta.mat not found: {META_MAT}")
    meta = scipy.io.loadmat(str(META_MAT), squeeze_me=True)
    synsets = meta["synsets"]
    wnids: dict[int, str] = {}
    for row in synsets:
        ilsvrc_id = int(row["ILSVRC2012_ID"])
        wnid = str(row["WNID"])
        num_children = int(row["num_children"])
        if num_children == 0 and 1 <= ilsvrc_id <= 1000:
            wnids[ilsvrc_id] = wnid
    if len(wnids) != 1000:
        raise RuntimeError(
            f"expected 1000 leaf synsets in meta.mat, got {len(wnids)}"
        )
    return [""] + [wnids[i] for i in range(1, 1001)]


def read_val_labels() -> list[int]:
    """Return the 50 000 1-based class ids from the val ground-truth file."""
    if not VAL_GT.exists():
        raise FileNotFoundError(f"val ground-truth not found: {VAL_GT}")
    with open(VAL_GT) as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    if len(labels) != 50000:
        raise RuntimeError(f"expected 50000 val labels, got {len(labels)}")
    return labels


def wnid_dir_done(d: Path, expected: int | None = None) -> bool:
    """True if the wnid dir has been fully populated."""
    sentinel = d / ".extract_done"
    if sentinel.exists():
        return True
    if expected is not None and d.is_dir():
        n = sum(1 for _ in d.iterdir() if _.suffix == ".JPEG")
        if n == expected:
            sentinel.touch()
            return True
    return False


def extract_train(out_root: Path, dry_run: bool) -> None:
    """Stream-extract the tar-of-tars. Each inner tar is one wnid."""
    out_root.mkdir(parents=True, exist_ok=True)
    if not TRAIN_TAR.exists():
        raise FileNotFoundError(f"train tar not found: {TRAIN_TAR}")

    with tarfile.open(TRAIN_TAR, "r") as outer:
        for i, member in enumerate(outer):
            if not member.name.endswith(".tar"):
                continue
            wnid = Path(member.name).stem  # e.g. n01440764
            wnid_dir = out_root / wnid
            if wnid_dir_done(wnid_dir):
                if i % 100 == 0:
                    print(f"[skip] {wnid} already extracted", flush=True)
                continue
            wnid_dir.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"[dry] would extract {wnid}")
                continue
            inner_buf = outer.extractfile(member)
            if inner_buf is None:
                print(f"[warn] {wnid}: extractfile returned None, skipping")
                continue
            data = inner_buf.read()
            with tarfile.open(fileobj=io.BytesIO(data), mode="r") as inner:
                inner.extractall(path=wnid_dir, filter="data")
            (wnid_dir / ".extract_done").touch()
            n = sum(1 for _ in wnid_dir.iterdir() if _.suffix == ".JPEG")
            print(f"[{i+1}/1000] {wnid}: {n} images", flush=True)


def extract_val(out_root: Path, dry_run: bool) -> None:
    """Extract flat val tar + sort into wnid subdirs per devkit ground-truth."""
    out_root.mkdir(parents=True, exist_ok=True)
    if not VAL_TAR.exists():
        raise FileNotFoundError(f"val tar not found: {VAL_TAR}")

    wnids = load_wnid_order()
    labels = read_val_labels()
    # val files are named ILSVRC2012_val_00000001.JPEG .. _00050000.JPEG
    # label[i] is the 1-based class id for ILSVRC2012_val_<i+1:08d>.JPEG
    for wnid in wnids[1:]:
        (out_root / wnid).mkdir(parents=True, exist_ok=True)

    with tarfile.open(VAL_TAR, "r") as tf:
        for member in tf:
            if not member.name.endswith(".JPEG"):
                continue
            stem = Path(member.name).stem  # ILSVRC2012_val_00000001
            try:
                idx = int(stem.split("_")[-1])
            except ValueError:
                print(f"[warn] could not parse val index from {member.name}")
                continue
            if not (1 <= idx <= 50000):
                print(f"[warn] val index {idx} out of range for {member.name}")
                continue
            class_id = labels[idx - 1]
            wnid = wnids[class_id]
            dest = out_root / wnid / member.name
            if dest.exists():
                continue
            if dry_run:
                continue
            buf = tf.extractfile(member)
            if buf is None:
                continue
            dest.write_bytes(buf.read())


def verify_counts(out_root: Path, expected: int, split_label: str) -> bool:
    n = sum(1 for _ in out_root.rglob("*.JPEG"))
    status = "OK" if n == expected else "MISMATCH"
    print(f"{split_label}: {n}/{expected} JPEGs on disk — {status}")
    return n == expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = imagenet_dir()
    print(f"ImageNet root: {root} ({'dry-run' if args.dry_run else 'live'})")

    if args.split in ("train", "all"):
        print("=== train (tar-of-tars, ~138 GB, 1000 inner tars) ===")
        extract_train(root / "train_full", args.dry_run)

    if args.split in ("val", "all"):
        print("=== val → test/ (Q2 resolution: ILSVRC2012 val is DCBM's test) ===")
        extract_val(root / "test", args.dry_run)

    if not args.dry_run:
        verify_counts(root / "train_full", 1281167, "train_full")
        verify_counts(root / "test", 50000, "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
