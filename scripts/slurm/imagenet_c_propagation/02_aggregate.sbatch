#!/bin/bash
# ImageNet Tier-2 aggregation — stitch 50 per-cell artifacts into one summary.
#
# CPU-only follow-up to 01_array.sbatch. Reads each cell's stage_metrics.jsonl
# + per_image dump, writes consolidated stage_metrics.jsonl + summary.json
# at the dataset root.
#
# Submit chained after the array:
#   sbatch --dependency=afterok:<arrayjobid> \
#       scripts/slurm/imagenet_c_propagation/02_aggregate.sbatch

#SBATCH --job-name=in1k-c-prop-agg
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet_c_propagation/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/imagenet_c_propagation"

srun python "${RUN_DIR}/aggregate.py"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
