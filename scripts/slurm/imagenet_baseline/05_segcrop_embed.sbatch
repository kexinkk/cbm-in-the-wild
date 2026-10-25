#!/bin/bash
# Step 5: CLIP-ViT-B/16 embed the SAM2 crops. Expected ~50 crops × 50 000
# images = ~2.5M crops (varies with SAM2's per-image mask count). At 140
# imgs/s batched, ~5 h on A100. Partition gpu_h100_short (72 h cap)
# replaces gpu_a100_il (2026-04-22) — a100_il queue is contested, h100_short
# has larger capacity and is ≥ as fast for CLIP forward.
#
# If cap is hit: segcrop_embed writes the .torch dict atomically at the
# end of the whole run (no per-split chunking). A hit-the-wall restart
# starts from scratch unless we add incremental-save support. If 24 h
# turns out insufficient, shard by crop-basename hash.
#
# Depends on step 04 (crops must exist).
#
# Submit:
#   sbatch --dependency=afterok:<job04> scripts/slurm/imagenet1000/05_segcrop_embed.sbatch
#
# Produces:
#   $DCBM_REPO/data/Segments/segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch

#SBATCH --job-name=in1k-segcrop
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

start_heartbeat 300 "${DCBM_REPO}/data/Segments"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/imagenet_segcrop_embed.py" \
    --batch-size 128

stop_heartbeat

echo "------------------------------------------------------------"
f="${DCBM_REPO}/data/Segments/segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch"
if [ -f "${f}" ]; then
    echo "segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch: $(stat -c %s "${f}") bytes"
else
    echo "segcrop_ImageNet_SAM2_CLIP-ViT-B16.torch: MISSING"
fi
echo "Finished: $(date -Iseconds)"
