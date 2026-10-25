#!/bin/bash
# Submit the ImageNet-1k pipeline with SLURM afterok dependencies.
#
# Stages 00..07: tar extract → split → subset → CLIP → SAM2 → segcrop →
# baseline sweep (array 0..4) → baseline replicate.
#
# Usage:
#   bash scripts/slurm/imagenet1000/submit_chain.sh [start_at]
#
#   start_at: stage number to begin from (00..07). Default 00.
#             Useful for resuming after a stage failed or an artifact was
#             manually corrected: e.g. `... submit_chain.sh 03` skips
#             extract + split + subset and starts from CLIP extract.
#             Use `06` to run only the baseline after the data-prep
#             artifacts are already on disk from a prior chain.
#
# Exits with non-zero if any `sbatch --parsable` call fails. Prints a
# summary of dispatched job IDs at the end — feed them to `watch.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
CHAIN_DIR="scripts/slurm/imagenet1000"

START="${1:-00}"
declare -A JOBS=()

submit_stage() {
    local stage="$1"
    local script="$2"
    local deps="$3"        # colon-separated job ids, or empty
    local dep_flag=""
    if [ -n "${deps}" ]; then
        dep_flag="--dependency=afterok:${deps}"
    fi
    local jid
    jid=$(sbatch --parsable ${dep_flag} "${CHAIN_DIR}/${script}")
    JOBS["${stage}"]="${jid}"
    echo "  ${stage}  job ${jid}  (${script})${deps:+  depends on ${deps}}"
    echo "${jid}"
}

echo "============================================================"
echo "Dispatching ImageNet chain from stage ${START}"
echo "Repo:   ${REPO_ROOT}"
echo "Chain:  ${CHAIN_DIR}"
echo "============================================================"

prev=""

stages=(
    "00:00_extract_tars.sbatch"
    "01:01_split_val.sbatch"
    "02:02_build_subsets.sbatch"
    "03:03_clip_extract.sbatch"
    "04:04_sam2_extract.sbatch"
    "05:05_segcrop_embed.sbatch"
    "06:06_baseline_sweep.sbatch"
    "07:07_baseline_replicate.sbatch"
)

# Stage-dependency graph (non-linear — 03 needs 01, 04 needs 02, 05 needs 04,
# 03 and 04 can run in parallel after their prereqs, 06 needs both CLIP
# embeddings (03) and segcrop embeddings (05)).
#
# Value is space-separated list of prereq STAGE IDs. At dispatch time we
# look up each stage's submitted job ID and colon-join them for SLURM's
# --dependency=afterok:X:Y syntax.
declare -A DEPS_FOR=(
    ["00"]=""
    ["01"]="00"
    ["02"]="01"
    ["03"]="01"
    ["04"]="02"
    ["05"]="04"
    ["06"]="03 05"
    ["07"]="06"
)

for entry in "${stages[@]}"; do
    stage="${entry%%:*}"
    script="${entry##*:}"
    if [[ "${stage}" < "${START}" ]]; then
        continue
    fi

    # Resolve prereq stages (space-separated) to their submitted job IDs;
    # colon-join for SLURM's multi-dep syntax.
    dep_stages="${DEPS_FOR[${stage}]}"
    dep_jids=""
    for ds in ${dep_stages}; do
        if [ -n "${JOBS[${ds}]:-}" ]; then
            dep_jids="${dep_jids:+${dep_jids}:}${JOBS[${ds}]}"
        fi
    done

    submit_stage "${stage}" "${script}" "${dep_jids}" > /dev/null
done

echo "============================================================"
echo "Submitted:"
for stage in "${!JOBS[@]}"; do
    printf "  stage %s -> %s\n" "${stage}" "${JOBS[${stage}]}"
done | sort

# Write job IDs to a file for watch.sh to pick up.
mkdir -p logs/slurm
IDS_FILE="logs/slurm/imagenet_chain_$(date +%Y%m%d_%H%M%S).ids"
{
    echo "# ImageNet chain submitted $(date -Iseconds)"
    for stage in "${!JOBS[@]}"; do
        echo "${stage} ${JOBS[${stage}]}"
    done | sort
} > "${IDS_FILE}"
echo "============================================================"
echo "Job IDs saved to ${IDS_FILE}"
echo "Monitor with:   bash ${CHAIN_DIR}/watch.sh"
echo "============================================================"
