#!/bin/bash
# ImageNet selective re-run with --dump-stage-tensors enabled.
#
# Re-runs 2 cells (gaussian_noise s5 = task 9, fog s5 = task 19) with the
# per-image clip_emb + concept_acts payload enriched. Existing JSONL/summary
# are byte-identical (metrics deterministic); per_image/*.pt is overwritten
# with the larger payload (≈ 500 MB/cell on imagenet = +1 GB total).
#
# Required by:
#   notebooks/corruption_propagation.ipynb §11 (cluster smear)
#   notebooks/corruption_propagation.ipynb §12 (rank-rank scatter)
#   per-concept robustness leaderboard 
#
# Submit:
#   sbatch scripts/slurm/imagenet_c_propagation/03_dump_selected.sbatch
# Reads: same as 01_array.sbatch.
# Writes: experiments/imagenet_c_propagation/{gaussian_noise_s5,fog_s5}/
#         per_image/<corruption>_s5.pt  (enriched payload)

#SBATCH --job-name=in1k-c-dump
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=9,19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0:28:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet_c_propagation/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/imagenet_c_propagation"

srun python "${RUN_DIR}/run_cell.py" --task-id "${SLURM_ARRAY_TASK_ID}" --dump-stage-tensors

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
