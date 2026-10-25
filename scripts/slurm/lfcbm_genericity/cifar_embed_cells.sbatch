#!/bin/bash
# CIFAR-10 corrupted test-split embedding cells (50 = 10 fam × 5 sev).
# Skip-if-exists per cell → safe to re-submit for stragglers.
# Reads:  data/raw/CIFAR-10-C/*.npy (bulk), cifar10 clean test via harness
# Writes: experiments/cifar10_lfcbm_genericity/corrupted_test/<fam>_s<sev>.pt

#SBATCH --job-name=lfcbm-c10-cells
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/lfcbm_genericity/_common.sh"
export LFCBM_REPO="${LFCBM_REPO:-${HOME}/Label-free-CBM}"

srun python experiments/cifar10_lfcbm_genericity/embed_corrupted_test.py \
    --task-id "${SLURM_ARRAY_TASK_ID}"
