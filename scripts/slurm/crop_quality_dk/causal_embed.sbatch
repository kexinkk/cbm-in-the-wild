#!/bin/bash
# §3.4.3b (T1b) causal test — GPU embed phase (decode+downsample member crops → centroids).
#
# Depends on the T1a run (membership.npz). Bounded work: <=3*R*MAX_MEMBERS crop decodes
# + CLIP embeds (default 50 concepts/arm, 200 members cap → <=30k images). Writes
# centroids_causal.npz; the d_k recompute runs separately on CPU (causal_dk.sbatch).
#
# Submit (after T1a finishes):
#   sbatch --dependency=afterok:<T1a_jobid> scripts/slurm/crop_quality_dk/causal_embed.sbatch
#
# 30-min short-partition cap: embed is fast (~min on A100); decode dominates. If it does
# not fit, lower MAX_MEMBERS or R, or move to gpu_a100_il.

#SBATCH --job-name=cq-causal-embed
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

srun python experiments/imagenet_crop_quality_dk/run_causal.py --phase embed

echo "------------------------------------------------------------"
echo "Artifacts: experiments/imagenet_crop_quality_dk/centroids_causal.npz"
echo "Finished:  $(date -Iseconds)"
