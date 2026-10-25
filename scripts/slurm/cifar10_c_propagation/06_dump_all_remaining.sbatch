#!/bin/bash
# CIFAR-10 full re-run with --dump-stage-tensors for the 45 cells that lack
# per-image embeddings (evaluation refactor, 2026-09-09: recovery grid moves
# to full-test scoring; CIFAR needs clip_emb + concept_acts in every cell).
#
# Skips the 5 cells already enriched by 03/04 (tasks 9, 19, 24, 39, 49 =
# gaussian_noise/fog/random_pixel_dropout/patch_masking/structured_pixel_dropout
# at s5). Existing JSONL/summary are byte-identical (metrics deterministic);
# per_image/*.pt is overwritten with the larger payload (~28 MB/cell, +1.3 GB).
#
# CIFAR inputs are monolithic (torchvision blobs + Hendrycks .npy) — no
# $TMPDIR staging needed; this is not a loose-tree job.
#
# Submit:
#   sbatch scripts/slurm/cifar10_c_propagation/06_dump_all_remaining.sbatch
# Writes: experiments/cifar10_c_propagation/<corruption>_s<sev>/per_image/*.pt

#SBATCH --job-name=cifar10-c-dumpall
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=0-8,10-18,20-23,25-38,40-48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10_c_propagation/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/cifar10_c_propagation"

srun python "${RUN_DIR}/run_cell.py" --task-id "${SLURM_ARRAY_TASK_ID}" --dump-stage-tensors

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
