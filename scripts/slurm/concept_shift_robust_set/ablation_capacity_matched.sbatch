#!/bin/bash
# §3.4.5 capacity-matched validation — native-K baseline (arm A) vs prune-from-2048 (arm B).
#
# Holds capacity fixed at each K_wanted and varies ONLY the concept selection, so the
# delta B − A under corruption is capacity-controlled evidence for the robust-set method.
# Both arms × {50,100,200,400,800,1200,1600,2048} × 3 seeds; native banks cached to
# experiments/.../capacity_matched/ (skip-if-exists → restartable, so a TIMEOUT resumes).
#
# GATE: only submit AFTER diag_shift_covariance.sbatch (§3.4.6) reports verdict_v2 STRUCTURED
# (the softmax-invariant residual — the v1 verdict gates on a uniform mode softmax cancels).
# If even the residual is one global mode, arm A ≈ arm B by construction.
#
# Submit (after the gate passes):
#   sbatch scripts/slurm/concept_shift_robust_set/ablation_capacity_matched.sbatch
#
# Knobs (env): RANK_BY=dk_hold3,dk,mk_perp,rand0,rand1,rand2 (arm-B rankings; rand<i> =
# selection-null control band — gate v2 predicts dk lands inside it),
# MAX_ITER=1000 (v1 runs at 200 left lbfgs unconverged — n_iter now logged per fit),
# N_INIT / CROP_SUBSAMPLE (KMeans cost), FIT_N=25000.
#
# Time: 24 native KMeans fits (arm A) + 7 arms x 8 sizes x 3 seeds = 168 logistic fits at
# MAX_ITER=1000 — the logistic side now rivals the KMeans side. 24h budgeted; BOTH sides are
# cached per cell (native_K*.pt + capacity_matched/results.jsonl), so a TIMEOUT re-submit
# (afterany chain) resumes where it stopped. Tune N_INIT=1 / CROP_SUBSAMPLE=200000 if long.

#SBATCH --job-name=cs-capmatch
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
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
# tune if the KMeans sweep runs long: N_INIT=1  CROP_SUBSAMPLE=200000
export N_INIT=${N_INIT:-3}

cd "${PROJECT_ROOT}"

srun python experiments/imagenet_concept_shift_robust_set/ablation_capacity_matched.py

echo "------------------------------------------------------------"
echo "Artifacts: experiments/imagenet_concept_shift_robust_set/ablation_capacity_matched.csv"
echo "Finished:  $(date -Iseconds)"
