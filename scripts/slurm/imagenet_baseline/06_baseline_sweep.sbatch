#!/bin/bash
# Step 6: ImageNet-1k + ViT-B/16 DCBM hyperparameter sweep, sharded across a
# SLURM array. Each task covers one cluster count × 3 (lambda, lr) cells
# (seed 0).
#
# If cap is hit: sweep writes sweep_shard_<N>.jsonl line-by-line as
# cells complete. A partially-written shard loses only in-progress
# cells on restart (the runner deletes the shard file at the start of
# run_sweep_cells and rewrites from scratch — this is a design weakness;
# if it ever matters, change the runner to append rather than truncate).
#
# Array = 0..4 matches the 5 cluster counts in config.sweep.n_clusters.
# Each task writes sweep_shard_<i>.jsonl.
#
# Submit chained after step 05 (segcrop embed):
#   sbatch --dependency=afterok:<job05> \
#       scripts/slurm/imagenet1000/06_baseline_sweep.sbatch
#
# Reads:
#   $DCBM_REPO/data/Embeddings/images_ImageNet_{train,val,test}_CLIP-ViT-B16.torch
#   $DCBM_REPO/data/Segments/segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch
#   $DCBM_REPO/data/Embeddings/subsets/ImageNet/ImageNet_rand_50.txt
# Writes:
#   experiments/imagenet_baseline/sweep_shard_<i>.jsonl

#SBATCH --job-name=in1k-sweep
#SBATCH --partition=gpu_h100
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/imagenet_baseline"

start_heartbeat 300 "${RUN_DIR}"

srun python "${RUN_DIR}/run.py" \
    --phase sweep \
    --cluster-index "${SLURM_ARRAY_TASK_ID}"

stop_heartbeat

echo "------------------------------------------------------------"
SHARD_LOG="${RUN_DIR}/sweep_shard_${SLURM_ARRAY_TASK_ID}.jsonl"
if [ -f "${SHARD_LOG}" ]; then
    echo "shard ${SLURM_ARRAY_TASK_ID}: $(wc -l < "${SHARD_LOG}") rows in sweep_shard_${SLURM_ARRAY_TASK_ID}.jsonl"
else
    echo "shard ${SLURM_ARRAY_TASK_ID}: MISSING sweep_shard_${SLURM_ARRAY_TASK_ID}.jsonl"
fi
echo "Finished: $(date -Iseconds)"
