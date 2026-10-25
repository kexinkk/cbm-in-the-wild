#!/bin/bash
# §3.4 Concept-shift robust set — compute runner (CPU-only).
#
# Streams all 50 ImageNet-C per-image cells (~25 GB torch.load) + clean ref to
# compute per-concept shift d_k, the confound diagnostic, robust/fragile sets,
# and the head-masking ablation. No GPU / CLIP / SAM2 — cached dumps only.
# Runs the standalone runner on a compute node (never the login node); the
# notebook is presentation-only and loads the artifacts this writes.
#
# Submit:
#   sbatch scripts/slurm/concept_shift_robust_set/run.sbatch

#SBATCH --job-name=concept-shift-robust
#SBATCH --partition=cpu,cpu_il
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
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

source /usr/share/lmod/lmod/init/profile 2>/dev/null || true
module purge
module load devel/miniforge/25.3.1-python-3.12
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"
which python; python --version

export PYTHONUNBUFFERED=1
# Keep BLAS from oversubscribing the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "${PROJECT_ROOT}"

srun python experiments/imagenet_concept_shift_robust_set/run.py

echo "------------------------------------------------------------"
echo "Artifacts: experiments/imagenet_concept_shift_robust_set/"
ls -la "${PROJECT_ROOT}/experiments/imagenet_concept_shift_robust_set/" || true
echo "Finished:  $(date -Iseconds)"
