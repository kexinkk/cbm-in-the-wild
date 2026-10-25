#!/bin/bash
# Create tar archives of ImageNet and CIFAR-10 in the workspace.
# These tars are used by extract_gpu_staged.sbatch to copy data to $TMPDIR
# (local NVMe) before GPU jobs, avoiding Lustre metadata storms.
#
# Usage:
#   sbatch -p cpu_il -t 6:00:00 scripts/slurm/data_prep/create_dataset_tars.sbatch
#
# Only creates tars that don't already exist. Safe to re-run.

#SBATCH --job-name=create-tars
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

WS=$(ws_find "${WS_NAME:-dcbm-data}" 2>/dev/null)
if [ -z "$WS" ]; then
    echo "ERROR: workspace dcbm-data not found"
    exit 1
fi

echo "Workspace: $WS"
echo "Started: $(date -Iseconds)"

# --- ImageNet ---
if [ -f "$WS/imagenet.tar" ]; then
    echo "[skip] imagenet.tar already exists ($(du -h "$WS/imagenet.tar" | cut -f1))"
else
    echo "[tar] creating imagenet.tar (~150GB, may take 1-2h)..."
    tar cf "$WS/imagenet.tar" -C "$WS" imagenet
    echo "[done] imagenet.tar ($(du -h "$WS/imagenet.tar" | cut -f1))"
fi

# --- CIFAR-10 ---
if [ -f "$WS/cifar10.tar" ]; then
    echo "[skip] cifar10.tar already exists ($(du -h "$WS/cifar10.tar" | cut -f1))"
else
    echo "[tar] creating cifar10.tar..."
    tar cf "$WS/cifar10.tar" -C "$WS" cifar10
    echo "[done] cifar10.tar ($(du -h "$WS/cifar10.tar" | cut -f1))"
fi

echo "Finished: $(date -Iseconds)"
