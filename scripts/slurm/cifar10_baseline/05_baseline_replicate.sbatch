#!/bin/bash
# Step 4b: Replicate the best-val sweep cell across 5 seeds. Reads every
# sweep_shard_*.jsonl (or sweep.jsonl), picks the argmax-val cell, runs
# seeds 0..4 back-to-back (~25 min on an A100), and writes summary.json
# with the pass/fail verdict vs target_mean_test_accuracy in the config.
#
# Submit chained after the sweep array:
#   sbatch --dependency=afterok:<sweeparrayid> \
#       scripts/slurm/cifar10/05_baseline_replicate.sbatch
#
# Reads: experiments/cifar10_baseline/sweep_shard_*.jsonl
# Writes:
#   experiments/cifar10_baseline/seeds.jsonl
#   experiments/cifar10_baseline/summary.json
#   experiments/cifar10_baseline/git_commit.txt

#SBATCH --job-name=cifar10-replicate
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/cifar10_baseline"

srun python "${RUN_DIR}/run.py" --phase replicate

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
