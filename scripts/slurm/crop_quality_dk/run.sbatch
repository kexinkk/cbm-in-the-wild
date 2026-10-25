#!/bin/bash
# §3.4.3b (T1a) crop-quality correlational pass — CPU-only.
#
# Reads per-crop JPEG headers (one metadata pass over the crops tree, cached to
# crop_dims.npz), assigns crops to the 2048 concepts (cosine argmax), and correlates
# per-concept median crop geometry (min_side / area / AR) with d_k. Writes membership.npz
# for the T1b causal runner.
#
# Submit:
#   sbatch scripts/slurm/crop_quality_dk/run.sbatch
#
# Time: the header pass over ~300k loose crops is metadata-bound (~20-40 min on Lustre);
# the analysis is fast. 2h is generous; cached crop_dims.npz makes reruns quick.

#SBATCH --job-name=cq-correlational
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
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
ls "${PROJECT_ROOT}/.conda/bin" >/dev/null 2>&1 || true
conda activate "${PROJECT_ROOT}/.conda"
which python; python --version

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

cd "${PROJECT_ROOT}"

srun python experiments/imagenet_crop_quality_dk/run.py

echo "------------------------------------------------------------"
echo "Artifacts: experiments/imagenet_crop_quality_dk/concept_quality_table.csv"
echo "Finished:  $(date -Iseconds)"
