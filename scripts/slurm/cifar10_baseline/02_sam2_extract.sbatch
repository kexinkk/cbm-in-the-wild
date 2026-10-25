#!/bin/bash
# Step 2: SAM2 segmentation on the 500-image CIFAR-10 subset, with the
# 32→1024 bicubic upsample monkey-patch. ~22 min on an A100 from scratch;
# resumes via `already_processed` if crops are already on disk.
#
# The subset file $DCBM_REPO/data/Embeddings/subsets/cifar10/cifar10_rand_50.txt
# must exist before this runs — produced by cifar10_build_subsets.py (local,
# ~1 s). Upstream ordering calls this "step 4 before step 6" but in practice
# step 6 is a prereq of step 4 (README, upstream DCBM notes).
#
# Submit (standalone):
#   sbatch scripts/slurm/cifar10/02_sam2_extract.sbatch
# or chained after CLIP extract:
#   sbatch --dependency=afterok:<clipjobid> scripts/slurm/cifar10/02_sam2_extract.sbatch
#
# Produces:
#   $DCBM_REPO/data/Segments/cifar10_SAM2/crops/<class>/<stem>_OUT_CROP_<i>_X_.jpg
#   $DCBM_REPO/data/Segments/stats/cifar10_SAM2_STATS.torch

#SBATCH --job-name=cifar10-sam2
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10/_common.sh"

# If the subset file is missing, build it first (cheap, CPU).
SUBSET="${HOME}/DCBM/data/Embeddings/subsets/cifar10/cifar10_rand_50.txt"
if [ ! -f "${SUBSET}" ]; then
    echo "[setup] subset file missing; building now"
    python "${SLURM_SUBMIT_DIR}/scripts/data_prep/cifar10_build_subsets.py"
fi

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/cifar10_sam2_extract.py"

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
