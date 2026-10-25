#!/bin/bash
# ImageNet Tier-2 corruption propagation array (50 cells).
#
# Each task evaluates one (corruption, severity) cell on the full 50k
# ImageNet test split through the deployed-DCBM forward path. Per-cell
# wall projection: ~6-7 min on H100, ~10-12 min on A100, plus ~90s setup
# (CLIP load + 250MB clean cache + class-name cache + corruptor build).
#
# Submit:
#   sbatch scripts/slurm/imagenet_c_propagation/01_array.sbatch
# Reads:
#   data/checkpoints/imagenet_baseline/{linear_head,centroids,scaler,manifest}*
#   data/checkpoints/imagenet_baseline/clean_propagation/*
#   data/imagenet/test/<wnid>/<image>.JPEG
#   data/ImageNet-C/<corruption>/<sev>/<wnid>/<image>.JPEG
# Writes:
#   experiments/imagenet_c_propagation/<corruption>_s<sev>/{stage_metrics.jsonl,
#                                                           summary.json,
#                                                           per_image/*.pt}

#SBATCH --job-name=in1k-c-prop
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0:25:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet_c_propagation/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/imagenet_c_propagation"

srun python "${RUN_DIR}/run_cell.py" --task-id "${SLURM_ARRAY_TASK_ID}"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
