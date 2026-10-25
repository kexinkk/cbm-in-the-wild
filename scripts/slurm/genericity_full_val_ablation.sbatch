#!/bin/bash
# Deployed-scale (full-val) capacity-matched ablation for the LF-CBM / PCBM
# genericity twins. Runs one resumable wave of a run_ablation.py living in a
# *_full_val experiment dir, on the 30-min short GPU partitions (GPU LBFGS
# logistic; no image model, no corrupted-image compute). afterany-safe:
# each wave stops before the wall (TIME_BUDGET_S) and resumes from
# <dir>/[<bank>_]ablation_results.jsonl.
#
#   sbatch scripts/slurm/genericity_full_val_ablation.sbatch \
#     experiments/imagenet_lfcbm_genericity_full_val/run_ablation.py
#   sbatch scripts/slurm/genericity_full_val_ablation.sbatch \
#     experiments/imagenet_pcbm_genericity_full_val/run_ablation.py --bank mmL

#SBATCH --job-name=gen-fv-abl
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0:25:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/lfcbm_genericity/_common.sh"

export TIME_BUDGET_S=1290
srun python "$@"
