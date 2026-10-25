#!/bin/bash
# ImageNet sensor-coverage re-run with --dump-stage-tensors enabled.
#
# Re-runs 3 sensor cells at sev=5 — chosen to span the sensor-family
# accuracy-drop range with one ranking anchor (rank-2 across both datasets):
#   task 24  random_pixel_dropout  asf=0.86  acc=0.23  (rank 2 on both DS)
#   task 29  salt_and_pepper       asf=0.88  acc=0.19  (imagenet sensor-high)
#   task 39  patch_masking         asf=0.66  acc=0.46  (sensor-low extreme)
#
# Selection criteria: PROPOSAL_visualization_0505.md §4.5 (revised).
# Companion to 03_dump_selected.sbatch (Hendrycks extremes: gaussian_noise,
# fog at sev=5).
#
# Submit:
#   sbatch scripts/slurm/imagenet_c_propagation/04_dump_sensor.sbatch
# Reads: same as 01_array.sbatch.
# Writes: experiments/imagenet_c_propagation/{random_pixel_dropout_s5,
#         salt_and_pepper_s5,patch_masking_s5}/per_image/*.pt
#         (enriched payload, ≈ 500 MB/cell)

#SBATCH --job-name=in1k-c-dump-sensor
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=24,29,39
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
