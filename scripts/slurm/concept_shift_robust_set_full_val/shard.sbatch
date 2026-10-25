#!/bin/bash
# One shard of the full-val capacity-matched grid: a single arm (or native) across all K/seeds,
# writing to its own OUT_DIR shard. OUT_DIR / RANK_BY / INCLUDE_NATIVE arrive from the submit
# script via --export; everything else is fixed here. The merge job concatenates the shards.
# ~18 fits/shard (6 K x 3 seeds) -> ~7 h; 24 h budgeted for headroom on the big-K fits.

#SBATCH --job-name=cmfv-shard
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname) @ $(date -Iseconds)"
echo "OUT_DIR=${OUT_DIR:?must be passed via --export}  RANK_BY='${RANK_BY-}'  INCLUDE_NATIVE=${INCLUDE_NATIVE:?}"

source /usr/share/lmod/lmod/init/profile 2>/dev/null || true
module purge
module load devel/miniforge/25.3.1-python-3.12
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
# fixed full-val config (OUT_DIR/RANK_BY/INCLUDE_NATIVE come from the job env)
export FIT_N=0
export SIZES="${SIZES:-100,400,800,1200,1600,K}"
export MAX_ITER="${MAX_ITER:-1000}"
export NATIVE_CACHE="${PROJECT_ROOT}/experiments/imagenet_concept_shift_robust_set/capacity_matched"
export N_INIT="${N_INIT:-3}"

cd "${PROJECT_ROOT}"
srun python experiments/imagenet_concept_shift_robust_set/ablation_capacity_matched.py
echo "Finished ${SLURM_JOB_NAME} @ $(date -Iseconds)  -> ${OUT_DIR}/capacity_matched/results.jsonl"
