#!/bin/bash
# =============================================================================
# SLURM array job template for per-image GPU work (SAM2, CLIP, corruptions).
#
# Usage:
#   sbatch --array=0-9 scripts/slurm/extract_gpu.sbatch \
#       python scripts/eval/run_sam2_extraction.py --dataset cifar10 --split test
#
# The --array=0-9 means 10 shards; each shard gets SHARD_ID in its environment.
# Your Python script reads SHARD_ID + SHARD_COUNT to process its slice.
#
# Config section below is the only part you normally edit per job type.
# =============================================================================

# --- SLURM directives ---
#SBATCH --job-name=dcbm-extract
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=190G
#SBATCH --time=3-00:00:00

# Output files: one per array task, includes both job and task id for traceability.
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# --- Fail fast on errors ---
set -euo pipefail

# --- Echo job metadata for debugging ---
echo "============================================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID:-not-an-array} of ${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started:    $(date -Iseconds)"
echo "Git commit: $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "============================================================"

# --- Load modules (compute nodes don't inherit login-node .bashrc) ---
module purge
module load devel/miniforge/25.3.1-python-3.12

# --- Initialize conda for non-interactive shells (required on compute nodes) ---
# bwHPC's miniforge module sets CONDA_EXE (path to the conda binary) but not
# CONDA_ROOT. The init script lives two levels up from the binary.
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

# --- Activate project env ---
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"

# Sanity: make sure the right Python is active.
which python
python --version

# --- Workspace path (same logic as ~/.bashrc, in case not inherited) ---
if command -v ws_find > /dev/null 2>&1; then
    export WS=$(ws_find "${WS_NAME:-dcbm-data}" 2>/dev/null)
    echo "Workspace: ${WS:-NOT FOUND}"
fi

# --- Shard environment variables for the Python script ---
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
export SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Processing shard ${SHARD_ID} of ${SHARD_COUNT}"

# --- The actual work: run whatever command was passed in ---
# Everything after the sbatch script path on the command line lands in "$@".
# Example:
#   sbatch --array=0-9 extract_gpu.sbatch python scripts/eval/run_sam2.py --dataset cifar10
# runs:
#   python scripts/eval/run_sam2.py --dataset cifar10
# with SHARD_ID=0..9 exported in the environment.
if [ $# -eq 0 ]; then
    echo "ERROR: no command passed to sbatch."
    echo "Usage: sbatch [--array=...] extract_gpu.sbatch <python script and its args>"
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
