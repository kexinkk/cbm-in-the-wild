#!/bin/bash
# Step 1 of CIFAR-10 baseline extraction: CLIP-ViT-B/16 image embeddings
# for train / val / test. ~10 min on an A100 for 60k images.
#
# Submit:
#   sbatch scripts/slurm/cifar10/01_clip_extract.sbatch
#
# Produces (under $DCBM_REPO/data/Embeddings/):
#   images_cifar10_{train,val,test}_CLIP-ViT-B16.torch
# Skip-if-exists: re-running is safe, pass --overwrite to regenerate.

#SBATCH --job-name=cifar10-clip
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:25:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10/_common.sh"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/cifar10_clip_extract.py"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
