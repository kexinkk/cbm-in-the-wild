#!/bin/bash
# Generic PCBM-genericity stage job for CIFAR-10 (post-bank: cached CIFAR
# embeddings + reused cifar10_lfcbm corrupted cells, no Broden staging).
# Usage (via cifar submit chain):
#   sbatch cifar_stage.sbatch <runner.py> [extra args...]
# Runs experiments/cifar10_pcbm_genericity/<runner.py>. Skip-if-exists /
# resumable, so afterany re-submission after TIMEOUT is harmless.

#SBATCH --job-name=pcbm-c10
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
srun python "experiments/cifar10_pcbm_genericity/${RUNNER}" "$@"
