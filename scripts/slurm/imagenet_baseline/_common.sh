# Shared env bootstrap + progress-tracking helpers for imagenet1000/*.sbatch.
# Source from inside each sbatch after its #SBATCH directives.
#
# Differences from the CIFAR-10 _common.sh:
#   1. PYTHONUNBUFFERED=1 so long-running stages stream to .out live.
#   2. TQDM_MININTERVAL=5 and TQDM_NCOLS=100 so tqdm emits a progress
#      line every 5 s (the default carriage-return behaviour collapses
#      to one jammed line on SLURM where stdout isn't a TTY).
#   3. start_heartbeat / stop_heartbeat bash functions — emit a
#      `[heartbeat]` line every N seconds so `watch.sh` can show
#      liveness and rough ETA without polling squeue.

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

# bwHPC compute nodes ship Lua modulefiles; classic Tcl `module` silently
# no-ops. Sourcing Lmod init is idempotent.
source /usr/share/lmod/lmod/init/profile 2>/dev/null || true
module purge
module load devel/miniforge/25.3.1-python-3.12

# Initialize conda for non-interactive shells.
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
conda activate "${PROJECT_ROOT}/.conda"

which python
python --version

# Progress-tracking knobs for Python children.
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5
export TQDM_NCOLS=100

# Workspace path — in case .bashrc isn't inherited on compute nodes.
if command -v ws_find > /dev/null 2>&1; then
    export WS=$(ws_find "${WS_NAME:-dcbm-data}" 2>/dev/null || true)
    echo "Workspace: ${WS:-NOT FOUND}"
fi

# DCBM repo resolution — respected by every data_prep/*.py we call.
export DCBM_REPO="${DCBM_REPO:-${HOME}/DCBM}"
echo "DCBM_REPO:  ${DCBM_REPO}"

# -----------------------------------------------------------------------------
# Heartbeat helpers — run a background bash loop that prints status every N s.
# Use in a stage sbatch like:
#     start_heartbeat 300 "$(pwd)"    # every 5 min, report cwd disk usage
#     python scripts/data_prep/imagenet_to_imagefolder.py
#     stop_heartbeat
# -----------------------------------------------------------------------------

_HEARTBEAT_PID=""

start_heartbeat() {
    local interval="${1:-300}"
    local watch_dir="${2:-}"
    (
        local t0=$(date +%s)
        while sleep "${interval}"; do
            local now=$(date +%s)
            local elapsed=$(( now - t0 ))
            local h=$(( elapsed / 3600 ))
            local m=$(( (elapsed % 3600) / 60 ))
            local disk=""
            if [ -n "${watch_dir}" ] && [ -d "${watch_dir}" ]; then
                disk=" files=$(find "${watch_dir}" -maxdepth 4 -type f 2>/dev/null | wc -l)"
            fi
            echo "[heartbeat t+${h}h${m}m${disk}] $(date -Iseconds)"
        done
    ) &
    _HEARTBEAT_PID=$!
    disown "${_HEARTBEAT_PID}" 2>/dev/null || true
    echo "[heartbeat started pid=${_HEARTBEAT_PID} interval=${interval}s]"
}

stop_heartbeat() {
    if [ -n "${_HEARTBEAT_PID}" ] && kill -0 "${_HEARTBEAT_PID}" 2>/dev/null; then
        kill "${_HEARTBEAT_PID}" 2>/dev/null || true
        wait "${_HEARTBEAT_PID}" 2>/dev/null || true
    fi
    _HEARTBEAT_PID=""
}

# Register trap so heartbeat doesn't survive an sbatch crash / timeout.
trap 'stop_heartbeat' EXIT

echo "------------------------------------------------------------"
