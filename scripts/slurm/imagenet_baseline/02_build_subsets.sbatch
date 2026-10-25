#!/bin/bash
# Step 2: build nested 5/10/25/50-per-class subset lists. Pure CPU.
# Must land before step 04 (SAM2) — the subset file is what SAM2 reads
# to resolve image paths.
#
# If cap is hit: build_subsets aborts if biggest output file already
# exists; partial output would trigger an abort on restart. Delete
# $DCBM_REPO/data/Embeddings/subsets/ImageNet/ and re-run.
#
# Submit (chained after step 01):
#   sbatch --dependency=afterok:<job01> scripts/slurm/imagenet1000/02_build_subsets.sbatch
#
# Produces:
#   $DCBM_REPO/data/Embeddings/subsets/ImageNet/ImageNet_rand_{5,10,25,50}.txt

# Routed to cpu_il: pure Python filesystem traversal + random sampling.

#SBATCH --job-name=in1k-subsets
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

srun python "${SLURM_SUBMIT_DIR}/scripts/data_prep/imagenet_build_subsets.py"

OUT="${DCBM_REPO}/data/Embeddings/subsets/ImageNet"
echo "------------------------------------------------------------"
for size in 5 10 25 50; do
    f="${OUT}/ImageNet_rand_${size}.txt"
    n=$(wc -l < "${f}" 2>/dev/null || echo "missing")
    echo "ImageNet_rand_${size}.txt: ${n} (expect $(( size * 1000 )))"
done
echo "Finished: $(date -Iseconds)"
