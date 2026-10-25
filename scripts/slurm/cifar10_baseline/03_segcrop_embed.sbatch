#!/bin/bash
# Step 3: CLIP-ViT-B/16 embed the SAM2 crops produced by step 2. Fast
# (~2-5 min on an A100 for a few thousand crops).
#
# Submit chained:
#   sbatch --dependency=afterok:<sam2jobid> \
#       scripts/slurm/cifar10/03_segcrop_embed.sbatch
#
# Produces the concept-bank file that CBM.load_concepts reads:
#   $DCBM_REPO/data/Segments/segcrop_cifar10_SAM2_CLIP-ViT-B16.torch

#SBATCH --job-name=cifar10-segcrop
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10/_common.sh"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/cifar10_segcrop_embed.py"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
