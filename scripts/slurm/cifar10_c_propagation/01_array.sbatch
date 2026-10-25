#!/bin/bash
# CIFAR-10 Tier-2 corruption propagation array (50 cells).
#
# Each task evaluates one (corruption, severity) cell on the full 10k
# CIFAR-10 test split through the deployed-DCBM forward path. Per-cell
# wall projection: ~30-40s on H100, ~60s on A100, plus ~60s setup
# (CLIP load + clean cache + class-name cache + corruptor build).
#
# Submit:
#   sbatch scripts/slurm/cifar10_c_propagation/01_array.sbatch
# Reads:
#   data/checkpoints/cifar10_baseline/{linear_head,centroids,scaler,manifest}*
#   data/checkpoints/cifar10_baseline/clean_propagation/*
#   data/cifar10/test/<class>/<image>.png
#   data/CIFAR-10-C/<corruption>.npy
# Writes:
#   experiments/cifar10_c_propagation/<corruption>_s<sev>/{stage_metrics.jsonl,
#                                                          summary.json,
#                                                          per_image/*.pt}

#SBATCH --job-name=cifar10-c-prop
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10_c_propagation/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/cifar10_c_propagation"

srun python "${RUN_DIR}/run_cell.py" --task-id "${SLURM_ARRAY_TASK_ID}"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
