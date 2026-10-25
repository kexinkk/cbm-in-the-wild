#!/bin/bash
# ImageNet scaled stage-2 metrics — array of 50 cells (~7 min each on H100).
#
# Recomputes argmax_shift, spearman_rho, top5_jaccard on StandardScaler-
# transformed concept activations (what the linear head reads).
#
# Submit:
#   sbatch scripts/slurm/imagenet_c_propagation/05_scaled_metrics.sbatch
# Aggregate after all tasks finish:
#   python experiments/imagenet_c_propagation/aggregate_scaled.py

#SBATCH --job-name=in1k-scaled-s2
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet_c_propagation/_common.sh"

srun python "${SLURM_SUBMIT_DIR}/experiments/imagenet_c_propagation/run_scaled_metrics.py" \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
