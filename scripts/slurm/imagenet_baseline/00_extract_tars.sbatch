#!/bin/bash
# Step 0: extract ILSVRC2012_img_train.tar + ILSVRC2012_img_val.tar into the
# ImageFolder layout. Heavy disk I/O, zero GPU use — routed to `cpu_il` for
# two reasons: (a) not wasting a GPU allocation, (b) the cpu_il pool (264
# nodes) is far less contested than gpu_a100_il, so dispatch latency drops
# from hours to minutes.
#
# Time cap: 12 h vs ~4-6 h realistic wall-clock (2-3x margin). Measured
# 2026-04-21: ~3.8 wnids/min on cpu_il uc2n838, projecting ~4.4 h for
# 1000 train wnids + ~5 min for the val tar. Original 2 h cap was wrong
# (based on a too-optimistic estimate); hit the wall at 39 min / 114 wnids.
# 12 h leaves generous slack for I/O contention or a slower node.
#
# If cap is hit: completed wnids have .extract_done sentinels; restart
# skips them. One wnid at most is re-extracted. Next run picks up where
# this one stopped.
#
# Submit:
#   sbatch scripts/slurm/imagenet1000/00_extract_tars.sbatch
#
# Produces:
#   $DCBM_DATA_DIR/imagenet/train_full/<wnid>/*.JPEG   (1 281 167 files)
#   $DCBM_DATA_DIR/imagenet/test/<wnid>/*.JPEG         (50 000 files)

#SBATCH --job-name=in1k-extract
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

# Watch the target dir so heartbeat can report file count as progress proxy.
TARGET="${SLURM_SUBMIT_DIR}/data/imagenet"
mkdir -p "${TARGET}"
start_heartbeat 300 "${TARGET}"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/imagenet_to_imagefolder.py"

stop_heartbeat

# Final count so summary.out rows make it easy to diff against expected.
echo "------------------------------------------------------------"
echo "train_full: $(find "${TARGET}/train_full" -name '*.JPEG' 2>/dev/null | wc -l) JPEGs (expect 1281167)"
echo "test:       $(find "${TARGET}/test" -name '*.JPEG' 2>/dev/null | wc -l) JPEGs (expect 50000)"
echo "Finished:   $(date -Iseconds)"
