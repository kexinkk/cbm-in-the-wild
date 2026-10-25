#!/bin/bash
# CIFAR-10 scaled stage-2 metrics — all 50 cells in one job (~5 min on H100).
#
# Recomputes argmax_shift, spearman_rho, top5_jaccard on StandardScaler-
# transformed concept activations (what the linear head reads).

#SBATCH --job-name=cifar10-scaled-s2
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10_c_propagation/_common.sh"

srun python "${SLURM_SUBMIT_DIR}/experiments/cifar10_c_propagation/run_scaled_metrics.py" \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
