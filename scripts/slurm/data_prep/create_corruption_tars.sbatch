#!/bin/bash
# Create per-corruption tar archives of ImageNet-C in the workspace.
#
# One tar per selected corruption (~2-7 GB each) so that gpu_a100_short
# cells can stage only the corruption they need to $TMPDIR in seconds,
# avoiding the 250K-file-per-corruption Lustre metadata storm.
#
# Output: $WS/raw/ImageNet-C/_tars/imagenet_c_<corruption>.tar
#
# CIFAR-10-C is .npy files (bulk reads, no metadata storm) — no tars needed.
#
# Usage:
#   sbatch scripts/slurm/data_prep/create_corruption_tars.sbatch
#
# Safe to re-run: skips existing tars.

#SBATCH --job-name=corr-tars
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

WS=$(ws_find "${WS_NAME:-dcbm-data}" 2>/dev/null)
if [ -z "$WS" ]; then
    echo "ERROR: workspace dcbm-data not found"
    exit 1
fi

SRC="$WS/raw/ImageNet-C"
TAR_DIR="$SRC/_tars"
mkdir -p "$TAR_DIR"

echo "Workspace: $WS"
echo "Source:    $SRC"
echo "Tar dir:  $TAR_DIR"
echo "Started:  $(date -Iseconds)"

CORRUPTIONS="fog gaussian_noise jpeg_compression motion_blur"

for corr in $CORRUPTIONS; do
    out="$TAR_DIR/imagenet_c_${corr}.tar"
    if [ -f "$out" ]; then
        echo "[skip] $(basename "$out") already exists ($(du -h "$out" | cut -f1))"
        continue
    fi
    if [ ! -d "$SRC/$corr" ]; then
        echo "[warn] $SRC/$corr not found, skipping"
        continue
    fi
    echo "[tar] $corr → $(basename "$out") ..."
    t0=$(date +%s)
    tar cf "$out" -C "$SRC" "$corr"
    t1=$(date +%s)
    echo "[done] $(basename "$out") ($(du -h "$out" | cut -f1)) in $((t1 - t0))s"
done

echo ""
echo "All tars:"
ls -lh "$TAR_DIR"/imagenet_c_*.tar 2>/dev/null || echo "  (none)"
echo ""
echo "Finished: $(date -Iseconds)"
