#!/bin/bash
# CIFAR-10 sensor-coverage re-run with --dump-stage-tensors enabled.
#
# Re-runs 3 sensor cells at sev=5 — chosen to span the sensor-family
# accuracy-drop range with one ranking anchor (rank-2 across both datasets):
#   task 24  random_pixel_dropout    asf=0.91  acc=0.40  (rank 2 on both DS)
#   task 39  patch_masking           asf=0.82  acc=0.62  (sensor-low extreme)
#   task 49  structured_pixel_dropout asf=0.95 acc=0.23  (cifar sensor-high)
#
# Selection criteria: PROPOSAL_visualization_0505.md §4.5 (revised).
# Companion to 03_dump_selected.sbatch (Hendrycks extremes: gaussian_noise,
# fog at sev=5).
#
# Submit:
#   sbatch scripts/slurm/cifar10_c_propagation/04_dump_sensor.sbatch
# Reads: same as 01_array.sbatch.
# Writes: experiments/cifar10_c_propagation/{random_pixel_dropout_s5,
#         patch_masking_s5,structured_pixel_dropout_s5}/per_image/*.pt
#         (enriched payload, ≈ 28 MB/cell)

#SBATCH --job-name=cifar10-c-dump-sensor
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=24,39,49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:12:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/cifar10_c_propagation/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/cifar10_c_propagation"

srun python "${RUN_DIR}/run_cell.py" --task-id "${SLURM_ARRAY_TASK_ID}" --dump-stage-tensors

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
