#!/bin/bash
# §3.2 full-val capacity-matched refit — head fit on the FULL 128,118-img val pool.
#
# Same runner as the 25k screen (ablation_capacity_matched.py), driven by env:
#   FIT_N=0          -> full val pool (effective N stamped as fit_n per result row)
#   OUT_DIR          -> results/CSV/summary land in the _full_val experiment dir
#   NATIVE_CACHE     -> reuse the original cached KMeans banks (fit_n-independent; no rebuild)
#   SIZES / RANK_BY  -> the 6-K grid and the 7 arm rankings incl. the new geom_ar
# Per-cell cache (native_K*.pt reused + OUT_DIR/capacity_matched/results.jsonl) makes this
# restartable: a TIMEOUT re-submit (afterany, via submit.sh) resumes where it stopped.
#
#   sbatch scripts/slurm/concept_shift_robust_set_full_val/run.sbatch
#   (or: bash scripts/slurm/concept_shift_robust_set_full_val/submit.sh  — chains resume waves)
#
# Time: ~126 logistic fits at full-val (K=2048 ~10-15 min each) -> ~12-16 h. 24h budgeted.

#SBATCH --job-name=cs-capmatch-fv
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "Started:    $(date -Iseconds)"
echo "Git commit: $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "============================================================"

source /usr/share/lmod/lmod/init/profile 2>/dev/null || true
module purge
module load devel/miniforge/25.3.1-python-3.12
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"
which python; python --version

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

# --- full-val configuration ---
export FIT_N=0
export SIZES="${SIZES:-100,400,800,1200,1600,K}"
export RANK_BY="${RANK_BY:-dk_hold3,rand0,rand1,rand2,geom,dkres_geom,geom_ar}"
export MAX_ITER="${MAX_ITER:-1000}"
export OUT_DIR="${PROJECT_ROOT}/experiments/imagenet_concept_shift_robust_set_full_val"
export NATIVE_CACHE="${PROJECT_ROOT}/experiments/imagenet_concept_shift_robust_set/capacity_matched"
export N_INIT="${N_INIT:-3}"   # native banks are cached, so this rarely fires

cd "${PROJECT_ROOT}"

srun python experiments/imagenet_concept_shift_robust_set/ablation_capacity_matched.py

echo "------------------------------------------------------------"
echo "Artifacts: experiments/imagenet_concept_shift_robust_set_full_val/ablation_capacity_matched.csv"
echo "Finished:  $(date -Iseconds)"
