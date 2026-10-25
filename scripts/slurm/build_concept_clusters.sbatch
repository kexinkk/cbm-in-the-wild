#!/bin/bash
# =============================================================================
# CPU-only SLURM job: reads a command from "$@" and runs it (no GPU).
# For light data-prep that loads heavy artifacts but does no GPU work — e.g.
# concept-cluster membership from the 688MB ImageNet segcrop blob.
#
# Usage:
#   sbatch scripts/slurm/build_concept_clusters.sbatch \
#       python scripts/data_prep/build_concept_clusters.py --dataset imagenet
# =============================================================================

# --- SLURM directives ---
#SBATCH --job-name=dcbm-clusters
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "Started:    $(date -Iseconds)"
echo "Git commit: $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "============================================================"

module purge
module load devel/miniforge/25.3.1-python-3.12
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
conda activate "${SLURM_SUBMIT_DIR}/.conda"
which python && python --version

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
