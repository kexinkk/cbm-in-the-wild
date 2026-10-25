#!/bin/bash
# §2.2.1 (T1b rep-crop) — n=1 causal probe, GPU embed phase.
#
# Naive single-crop variant of causal_embed.sbatch: degrade/re-embed ONLY each concept's
# representative (medoid) crop instead of the ~200-member cluster mean. Bounded work:
# 3*R crop decodes + CLIP embeds (default R=50 → 150 images) plus one crop-blob load to
# recover the medoid indices. Writes centroids_repcrop.npz; d_k recompute is separate CPU
# (causal_repcrop_dk.sbatch). Depends on the T1a run (membership.npz).
#
# Submit:
#   sbatch scripts/slurm/crop_quality_dk/causal_repcrop_embed.sbatch          # membership.npz exists
#   sbatch --dependency=afterok:<T1a> scripts/slurm/crop_quality_dk/causal_repcrop_embed.sbatch

#SBATCH --job-name=cq-repcrop-embed
#SBATCH --partition=gpu_a100_short,gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-none}"
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

cd "${PROJECT_ROOT}"

srun python experiments/imagenet_crop_quality_dk/run_causal.py --phase embed --rep-crop

echo "------------------------------------------------------------"
echo "Artifacts: experiments/imagenet_crop_quality_dk/centroids_repcrop.npz"
echo "Finished:  $(date -Iseconds)"
