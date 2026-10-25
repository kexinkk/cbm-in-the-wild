#!/bin/bash
# Generic PCBM genericity stage job (post-bank: no Broden staging needed —
# these stages run on cached ImageNet embeddings + the reused imagenet_c
# clip_emb cells). Usage (via submit_all.sh):
#   sbatch stage.sbatch <runner.py> [extra args...]
# All stages are skip-if-exists / resumable, so afterany re-submission after
# TIMEOUT is harmless.

#SBATCH --job-name=pcbm
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0:25:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/pcbm_genericity/_common.sh"

RUNNER="$1"; shift
export TIME_BUDGET_S=1290

srun python "experiments/imagenet_pcbm_genericity/${RUNNER}" "$@"
