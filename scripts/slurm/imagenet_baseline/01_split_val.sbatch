#!/bin/bash
# Step 1: 90/10 train/val internal split via symlinks. CPU + filesystem only.
# 1.28M symlinks on Lustre — metadata-bound.
#
# Ratio 0.10 is the committed default:
# overall split = 86.6/9.6/3.8 (train/internal-val/test), where test is ILSVRC2012 val (50k).
#
# If cap is hit: imagenet_split_val.py skips any symlink that already exists (and is a symlink to the correct target). Safe to re-run.
#
# Submit (chained after step 00):
#   sbatch --dependency=afterok:<job00> scripts/slurm/imagenet1000/01_split_val.sbatch
#
# Produces:
#   $DCBM_DATA_DIR/imagenet/train/<wnid>/*.JPEG   (~1.153M symlinks, 90%)
#   $DCBM_DATA_DIR/imagenet/val/<wnid>/*.JPEG     (~128k symlinks, 10%)

# Routed to cpu_il: pure filesystem (symlinking) work, no GPU needed.
# 3-day cap on cpu_il is wildly over-budget but gets us into the less-
# contested CPU pool — typical dispatch within minutes.

#SBATCH --job-name=in1k-split
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

start_heartbeat 60 "${SLURM_SUBMIT_DIR}/data/imagenet/train"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/imagenet_split_val.py"

stop_heartbeat

TARGET="${SLURM_SUBMIT_DIR}/data/imagenet"
echo "------------------------------------------------------------"
echo "train (symlinks): $(find "${TARGET}/train" -name '*.JPEG' 2>/dev/null | wc -l)"
echo "val   (symlinks): $(find "${TARGET}/val" -name '*.JPEG' 2>/dev/null | wc -l)"
echo "Finished:         $(date -Iseconds)"
