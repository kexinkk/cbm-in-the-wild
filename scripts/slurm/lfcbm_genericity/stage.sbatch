#!/bin/bash
# Generic LF-CBM genericity stage job. Usage (via submit_all.sh):
#   sbatch stage.sbatch <dataset> <runner.py> [extra args...]
# Runs experiments/<dataset>_lfcbm_genericity/<runner.py> on a short GPU
# partition. All stages are skip-if-exists / state-resumable, so afterany
# re-submission after TIMEOUT is harmless.

#SBATCH --job-name=lfcbm
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0:25:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/lfcbm_genericity/_common.sh"

DATASET="$1"; RUNNER="$2"; shift 2
export LFCBM_REPO="${LFCBM_REPO:-${HOME}/Label-free-CBM}"
export TIME_BUDGET_S=1290

srun python "experiments/${DATASET}_lfcbm_genericity/${RUNNER}" "$@"
