#!/bin/bash
# =============================================================================
# GPU job template with $TMPDIR staging for corruption evaluation cells.
#
# Stages up to TWO tars to local NVMe before running the Python command:
#   1. Clean dataset tar  (DCBM_STAGE_DATASET — same as extract_gpu_staged.sbatch)
#   2. Corruption tar     (DCBM_STAGE_CORRUPTION — optional, for Hendrycks cells)
#
# After staging, two env vars point Python at the local copies:
#   DCBM_DATA_DIR    → $TMPDIR  (clean dataset)
#   DCBM_IMGNET_C_DIR → $TMPDIR/ImageNet-C  (Hendrycks corruptions, if staged)
#
# Sensor corruptions don't need a corruption tar — they read the clean images
# (already staged) and apply transforms in memory.
#
# Usage examples:
#
#   # Hendrycks cell (stages clean + corruption tar):
#   DCBM_STAGE_DATASET=imagenet DCBM_STAGE_CORRUPTION=fog \
#       sbatch -p gpu_a100_short -t 0:28:00 \
#       scripts/slurm/extract_gpu_corruption.sbatch \
#       python experiments/imagenet_c_propagation/run_cell.py --corruption fog --severity 3
#
#   # Sensor cell (stages clean only):
#   DCBM_STAGE_DATASET=imagenet \
#       sbatch -p gpu_a100_short -t 0:28:00 \
#       scripts/slurm/extract_gpu_corruption.sbatch \
#       python experiments/imagenet_c_propagation/run_cell.py --corruption random_pixel_dropout --severity 2
#
#   # CIFAR-10 (stages clean; CIFAR-10-C is .npy — no corruption tar needed):
#   DCBM_STAGE_DATASET=cifar10 \
#       sbatch -p gpu_a100_short -t 0:28:00 \
#       scripts/slurm/extract_gpu_corruption.sbatch \
#       python experiments/cifar10_c_propagation/run_cell.py --corruption fog --severity 1
# =============================================================================

#SBATCH --job-name=dcbm-corr
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=0:28:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

# --- Parse env ---
DATASET="${DCBM_STAGE_DATASET:-}"
CORRUPTION="${DCBM_STAGE_CORRUPTION:-}"

if [ -z "$DATASET" ]; then
    echo "ERROR: set DCBM_STAGE_DATASET=imagenet|cifar10"
    exit 1
fi

# --- Job metadata ---
echo "============================================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID:-not-an-array} of ${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Dataset:    ${DATASET}"
echo "Corruption: ${CORRUPTION:-sensor (no tar)}"
echo "Started:    $(date -Iseconds)"
echo "Git commit: $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "============================================================"

# --- Load modules ---
module purge
module load devel/miniforge/25.3.1-python-3.12
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"
which python
python --version

# --- Workspace ---
if command -v ws_find > /dev/null 2>&1; then
    export WS=$(ws_find "${WS_NAME:-dcbm-data}" 2>/dev/null)
    echo "Workspace: ${WS:-NOT FOUND}"
fi

# --- Stage 1: clean dataset tar → $TMPDIR ---
TAR_PATH="${WS}/${DATASET}.tar"
if [ ! -f "$TAR_PATH" ]; then
    echo "ERROR: tar not found at $TAR_PATH"
    exit 1
fi

echo "Staging ${DATASET}.tar to \$TMPDIR ($(df -h "$TMPDIR" | tail -1 | awk '{print $4}') free)..."
t0=$(date +%s)
cp "$TAR_PATH" "$TMPDIR/"
tar xf "$TMPDIR/${DATASET}.tar" -C "$TMPDIR"
rm "$TMPDIR/${DATASET}.tar"
t1=$(date +%s)
echo "Staged clean dataset in $((t1 - t0))s"

export DCBM_DATA_DIR="$TMPDIR"
echo "DCBM_DATA_DIR=$DCBM_DATA_DIR"

# --- Stage 2: corruption tar → $TMPDIR (Hendrycks only) ---
if [ -n "$CORRUPTION" ]; then
    CORR_TAR="${WS}/raw/ImageNet-C/_tars/imagenet_c_${CORRUPTION}.tar"
    if [ ! -f "$CORR_TAR" ]; then
        echo "ERROR: corruption tar not found at $CORR_TAR"
        echo "Run: sbatch scripts/slurm/data_prep/create_corruption_tars.sbatch"
        exit 1
    fi
    mkdir -p "$TMPDIR/ImageNet-C"
    echo "Staging imagenet_c_${CORRUPTION}.tar ..."
    t0=$(date +%s)
    cp "$CORR_TAR" "$TMPDIR/"
    tar xf "$TMPDIR/imagenet_c_${CORRUPTION}.tar" -C "$TMPDIR/ImageNet-C"
    rm "$TMPDIR/imagenet_c_${CORRUPTION}.tar"
    t1=$(date +%s)
    echo "Staged corruption data in $((t1 - t0))s"

    export DCBM_IMGNET_C_DIR="$TMPDIR/ImageNet-C"
    echo "DCBM_IMGNET_C_DIR=$DCBM_IMGNET_C_DIR"
fi

# --- Shard env ---
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
export SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Processing shard ${SHARD_ID} of ${SHARD_COUNT}"

# --- Run the command ---
if [ $# -eq 0 ]; then
    echo "ERROR: no command passed to sbatch."
    exit 1
fi

echo "Running: $@"
echo "------------------------------------------------------------"
srun "$@"
exit_code=$?

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
echo "Exit code: ${exit_code}"
exit "${exit_code}"
