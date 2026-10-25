#!/bin/bash
# Step 4a: CIFAR-10 + ViT-B/16 DCBM hyperparameter sweep, sharded across a
# SLURM array. Each array task covers one cluster count × 3 (λ, lr) pairs
# (seed 0), roughly 15 min of training — fits comfortably under the 30-min
# gpu_a100_short cap.
#
# Array = 0..4 matches the 5 cluster counts in the config's sweep.n_clusters
# list. Each task writes sweep_shard_<i>.jsonl so shards don't race on one
# file; the replicate step merges them.
#
# Submit:
#   sbatch --dependency=afterok:<segcropjobid> \
#       scripts/slurm/cifar10/04_baseline_sweep.sbatch
#
# Reads:
#   $DCBM_REPO/data/Embeddings/images_cifar10_{train,val,test}_CLIP-ViT-B16.torch
#   $DCBM_REPO/data/Segments/segcrop_cifar10_SAM2_CLIP-ViT-B16.torch
#   $DCBM_REPO/data/Embeddings/subsets/cifar10/cifar10_rand_50.txt
# Writes:
#   experiments/cifar10_baseline/sweep_shard_<i>.jsonl

#SBATCH --job-name=cifar10-sweep
#SBATCH --partition=gpu_a100_short
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:25:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/cifar10_baseline"

srun python "${RUN_DIR}/run.py" \
    --phase sweep \
    --cluster-index "${SLURM_ARRAY_TASK_ID}"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
