#!/bin/bash
# =============================================================================
# GPU job template with $TMPDIR dataset staging.
#
# Same as extract_gpu.sbatch, but copies a dataset tar to local NVMe first,
# then sets DCBM_DATA_DIR so Python reads from fast local storage instead of
# Lustre (avoids metadata storms / admin warnings).
#
# Usage:
#   sbatch -p gpu_h100 -t 5:00:00 --dataset=imagenet \
#       scripts/slurm/extract_gpu_staged.sbatch \
#       python scripts/data_prep/imagenet_clip_extract.py --model RN50
#
# The --dataset flag is consumed by this script (not passed to Python).
# Supported values: imagenet, cifar10
# =============================================================================

#SBATCH --job-name=dcbm-staged
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=190G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

# --- Parse --dataset from SBATCH comment or environment ---
DATASET="${DCBM_STAGE_DATASET:-}"
if [ -z "$DATASET" ]; then
    echo "ERROR: set DCBM_STAGE_DATASET=imagenet|cifar10 in your sbatch call."
    echo "  Example: DCBM_STAGE_DATASET=imagenet sbatch ... extract_gpu_staged.sbatch ..."
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

# --- Stage dataset tar to $TMPDIR ---
TAR_PATH="${WS}/${DATASET}.tar"
if [ ! -f "$TAR_PATH" ]; then
    echo "ERROR: tar not found at $TAR_PATH"
    echo "Run: sbatch -p cpu_il -t 6:00:00 scripts/slurm/data_prep/create_dataset_tars.sbatch"
    exit 1
fi

echo "Staging ${DATASET}.tar to \$TMPDIR ($(df -h "$TMPDIR" | tail -1 | awk '{print $4}') free)..."
t_stage_start=$(date +%s)
cp "$TAR_PATH" "$TMPDIR/"
echo "Extracting..."
tar xf "$TMPDIR/${DATASET}.tar" -C "$TMPDIR"
rm "$TMPDIR/${DATASET}.tar"
t_stage_end=$(date +%s)
echo "Staged in $((t_stage_end - t_stage_start))s. Contents:"
ls "$TMPDIR/${DATASET}/" | head -5
echo "..."

# --- Override data dir so paths.py reads from local SSD ---
export DCBM_DATA_DIR="$TMPDIR"
echo "DCBM_DATA_DIR=$DCBM_DATA_DIR"

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
