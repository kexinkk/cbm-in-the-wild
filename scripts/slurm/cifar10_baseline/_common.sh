# Shared env bootstrap for cifar10/*.sbatch scripts. Source from inside each
# sbatch after its #SBATCH directives — not an sbatch itself.
#
# SLURM compute-node shells don't inherit the login-node profile, so every
# job must (a) initialize Lmod, (b) `module load` miniforge, (c) init conda,
# (d) activate the project env. Doing this once here keeps per-step sbatch
# scripts short and uniform.

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

# bwHPC compute nodes ship Lua modulefiles; the classic Tcl `module` wrapper
# silently no-ops on them. Sourcing Lmod init is idempotent.
source /usr/share/lmod/lmod/init/profile 2>/dev/null || true
module purge
module load devel/miniforge/25.3.1-python-3.12

# Initialize conda for non-interactive shells.
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"

which python
python --version

# Workspace path (in case .bashrc isn't inherited).
if command -v ws_find > /dev/null 2>&1; then
    export WS=$(ws_find "${WS_NAME:-dcbm-data}" 2>/dev/null || true)
    echo "Workspace: ${WS:-NOT FOUND}"
fi

echo "------------------------------------------------------------"
