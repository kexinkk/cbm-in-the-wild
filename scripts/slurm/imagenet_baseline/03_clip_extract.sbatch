#!/bin/bash
# Step 3: CLIP-ViT-B/16 embeddings for ImageNet train / val / test.
# H100 because 1.28M+128k+50k = ~1.46M images — the ~2x speedup over A100
# pays its queue cost.
#
# Depends on step 01 (train/val splits must exist) — test uses the tarball
# output from step 00, so technically this could run after step 00 if we
# skipped the train/val splits, but we always want all three.
#
# Time cap: 6 h vs ~1.5-2 h estimated on H100 (extrapolating from
# CIFAR-10's 140 imgs/s batch-of-64 throughput). 3-4x margin to absorb
# slower nodes, I/O contention on Lustre, or thermal throttling.
#
# If cap is hit: the script writes per-split output atomically via
# torch.save AFTER all images in that split are embedded. A partial run
# that hasn't finished a split yet loses its progress. To resume faster
# on retry, consider sharding per-split via env vars (not currently
# supported — add later if 6 h turns out insufficient).
#
# Submit:
#   sbatch --dependency=afterok:<job01> scripts/slurm/imagenet1000/03_clip_extract.sbatch
#
# Produces (in $DCBM_REPO/data/Embeddings/):
#   images_ImageNet_{train,val,test}_CLIP-ViT-B16.torch

#SBATCH --job-name=in1k-clip
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

start_heartbeat 300 "${DCBM_REPO}/data/Embeddings"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/imagenet_clip_extract.py" \
    --batch-size 64

stop_heartbeat

echo "------------------------------------------------------------"
for split in train val test; do
    f="${DCBM_REPO}/data/Embeddings/images_ImageNet_${split}_CLIP-ViT-B16.torch"
    if [ -f "${f}" ]; then
        size=$(stat -c %s "${f}")
        echo "images_ImageNet_${split}_CLIP-ViT-B16.torch: ${size} bytes"
    else
        echo "images_ImageNet_${split}_CLIP-ViT-B16.torch: MISSING"
    fi
done
echo "Finished: $(date -Iseconds)"
