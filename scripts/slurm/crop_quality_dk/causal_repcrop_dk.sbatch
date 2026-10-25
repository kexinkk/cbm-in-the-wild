#!/bin/bash
# §2.2.1 (T1b rep-crop) — n=1 causal probe, CPU d_k recompute phase.
#
# Reads centroids_repcrop.npz, streams the clean + corrupt cells once, recomputes d_k' for
# every treated n=1 centroid (per-concept independent → one pass), writes
# causal_dk_table_repcrop.csv + summary_causal_repcrop.json. Off the GPU short partition
# because the ~50-cell torch.load I/O far exceeds 30 min.
#
# Submit (after the rep-crop embed finishes):
#   sbatch --dependency=afterok:<embed_jobid> scripts/slurm/crop_quality_dk/causal_repcrop_dk.sbatch

#SBATCH --job-name=cq-repcrop-dk
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

cd "${PROJECT_ROOT}"

srun python experiments/imagenet_crop_quality_dk/run_causal.py --phase dk --rep-crop

echo "------------------------------------------------------------"
echo "Artifacts: experiments/imagenet_crop_quality_dk/causal_dk_table_repcrop.csv"
echo "Finished:  $(date -Iseconds)"
