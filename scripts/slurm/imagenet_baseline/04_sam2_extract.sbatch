#!/bin/bash
# Step 4: SAM2 on the 50 000-image concept subset, as a SLURM array of 4
# independent single-GPU shards (SHARD_ID = SLURM_ARRAY_TASK_ID ∈ {0,1,2,3}).
# Each shard processes every 4th image; per-image wall-clock ~4 s, so each
# shard is ~14 h on one H100 and the whole array completes in ~14 h wall
# (same as the prior in-node 4-GPU setup).
#
# Why array instead of single job with srun × 4 (rewrite 2026-04-22):
#   - One task-2 CUDA glitch in job 4078515 killed its srun task but left
#     the other 3 running, so the whole 04 job would have exited non-zero
#     and poisoned afterok:04 for stages 05/06/07 — DependencyNeverSatisfied
#     even though shards 0/1/3 finished cleanly. Array model isolates
#     failures: only the failing shard needs `scontrol requeue
#     <arrayjob>_<shard>` (or `sbatch --array=<shard>`) and the chain
#     resumes automatically once it succeeds.
#   - Each shard lands on its own node — no more shared-node
#     CUDA_VISIBLE_DEVICES ambiguity. Better backfill: 4 × (1 GPU, 8 CPU,
#     48 GB) packs into smaller gaps than 1 × (4 GPU, 32 CPU, 192 GB).
#
# If cap is hit: already_processed() skip-if-exists works per image
# (crops written to disk under crops/<wnid>/<stem>_OUT_CROP_<i>_X_.jpg).
# Restart resumes from the last completed image in the failing shard.
# Partial stats file per shard is updated in place.
#
# Each shard reads SHARD_ID = SLURM_ARRAY_TASK_ID and SHARD_COUNT = 4.
# Array job-level afterok dependency (used by stage 05) fires only when
# all 4 shards exit 0.
#
# Depends on step 02 (subset file must exist).
#
# Submit:
#   sbatch --dependency=afterok:<job02> scripts/slurm/imagenet1000/04_sam2_extract.sbatch
#
# Resubmit a single failed shard (no need to touch the others):
#   sbatch --array=2 scripts/slurm/imagenet1000/04_sam2_extract.sbatch
#   # or: scontrol requeue <arrayjob>_2
#
# Produces:
#   $DCBM_REPO/data/Segments/imagenet_SAM2/crops/<wnid>/*.jpg
#   $DCBM_REPO/data/Segments/stats/imagenet_SAM2_STATS_shard{0..3}of4.torch

#SBATCH --job-name=in1k-sam2
#SBATCH --partition=gpu_h100_short
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

# Heartbeat tracks the shared crops/ dir — every shard writes there.
CROPS_DIR="${DCBM_REPO}/data/Segments/imagenet_SAM2/crops"
mkdir -p "${CROPS_DIR}"
start_heartbeat 600 "${CROPS_DIR}"

export SHARD_ID="${SLURM_ARRAY_TASK_ID}"
export SHARD_COUNT=4
echo "[shard ${SHARD_ID}/${SHARD_COUNT}] node=$(hostname) cuda=${CUDA_VISIBLE_DEVICES}"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/imagenet_sam2_extract.py"

stop_heartbeat

echo "------------------------------------------------------------"
STATS_FILE="${DCBM_REPO}/data/Segments/stats/imagenet_SAM2_STATS_shard${SHARD_ID}of${SHARD_COUNT}.torch"
if [ -f "${STATS_FILE}" ]; then
    echo "shard ${SHARD_ID}: stats present ($(stat -c %s "${STATS_FILE}") bytes)"
else
    echo "shard ${SHARD_ID}: stats MISSING"
fi
echo "Finished: $(date -Iseconds)"
