#!/bin/bash
# =============================================================================
# Generic CPU-only SLURM wrapper (no GPU). Reads the command from "$@", exports
# SHARD_ID/SHARD_COUNT for array tasks, same contract as extract_gpu.sbatch.
#
# Usage:
#   sbatch --array=0-50 scripts/slurm/cpu_task.sbatch \
#       python experiments/<slug>/runner.py --cell-index '$SHARD_ID'
# =============================================================================

#SBATCH --job-name=dcbm-cpu
# Multi-partition: scheduler starts on whichever big CPU partition dispatches first.
# (dev_cpu / dev_cpu_il are 30-min, ≤8-node, and reject large arrays on QOS — add
#  them only for small single jobs via `sbatch --partition=...,dev_cpu`.)
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

echo "============================================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID:-not-an-array} of ${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "Started:    $(date -Iseconds)"
echo "Git commit: $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "============================================================"

module purge
module load devel/miniforge/25.3.1-python-3.12
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"
which python && python --version

# Keep BLAS threads matched to the allocation (avoids oversubscription).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
export SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

if [ $# -eq 0 ]; then
    echo "ERROR: no command passed to sbatch."
    exit 1
fi

echo "Running: $@"
echo "------------------------------------------------------------"
srun "$@"
exit_code=$?
echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)  exit=${exit_code}"
exit "${exit_code}"
