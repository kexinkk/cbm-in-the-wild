# Shared env bootstrap for imagenet_c_propagation/*.sbatch.
# Source from inside each sbatch after its #SBATCH directives.
#
# SLURM compute nodes don't inherit login-node profiles, so each job must
# (a) initialize Lmod, (b) load miniforge, (c) init conda, (d) activate the
# project env. Centralised here to keep per-stage sbatch scripts short.

set -euo pipefail

echo "============================================================"
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID:-not-an-array} of ${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started:    $(date -Iseconds)"
echo "Git commit: $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "============================================================"

source /usr/share/lmod/lmod/init/profile 2>/dev/null || true
module purge
module load devel/miniforge/25.3.1-python-3.12

source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"

which python
python --version

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5
export TQDM_NCOLS=100

if command -v ws_find > /dev/null 2>&1; then
    export WS=$(ws_find "${WS_NAME:-dcbm-data}" 2>/dev/null || true)
    echo "Workspace: ${WS:-NOT FOUND}"
fi

export DCBM_REPO="${DCBM_REPO:-${HOME}/DCBM}"
echo "DCBM_REPO:  ${DCBM_REPO}"

echo "------------------------------------------------------------"
