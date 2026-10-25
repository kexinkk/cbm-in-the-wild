#!/bin/bash
# Submit the full LF-CBM genericity chain for one dataset on the 30-min
# short GPU partitions, afterany-chained (timeouts are harmless: every
# stage is skip-if-exists or state-resumable).
#
#   bash scripts/slurm/lfcbm_genericity/submit_all.sh cifar10
#   bash scripts/slurm/lfcbm_genericity/submit_all.sh imagenet
#
# Chain: stage-A ─→ stage-B waves ─→ propagation
#                                └─→ concept-shift ─→ ablation waves
# cifar10 additionally runs the corrupted-cell embedding array in parallel
# with training; propagation/concept-shift wait on both.

set -euo pipefail
DATASET="${1:?usage: submit_all.sh <cifar10|imagenet>}"
cd "$(dirname "$0")/../../.."

SB=scripts/slurm/lfcbm_genericity/stage.sbatch
JOBS_FILE="experiments/${DATASET}_lfcbm_genericity/slurm_jobs.txt"

case "$DATASET" in
  cifar10)  B_WAVES=3; ABL_WAVES=2 ;;
  imagenet) B_WAVES=6; ABL_WAVES=8 ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

sub() {  # sub <name> <dep-or-empty> <args...>
  local name="$1" dep="$2"; shift 2
  local depflag=()
  [[ -n "$dep" ]] && depflag=(--dependency="afterany:${dep}")
  local id
  id=$(sbatch --parsable --job-name="$name" "${depflag[@]}" "$@")
  echo "${id} ${name}" | tee -a "$JOBS_FILE"
  LAST_ID="$id"
}

echo "# submitted $(date -Iseconds)" >> "$JOBS_FILE"

# stage A: concept layer
sub "lfcbm-${DATASET}-A" "" "$SB" "$DATASET" train_lfcbm.py --stage a
A_ID=$LAST_ID

# stage B: chained saga waves
DEP=$A_ID
for w in $(seq 1 "$B_WAVES"); do
  sub "lfcbm-${DATASET}-B${w}" "$DEP" "$SB" "$DATASET" train_lfcbm.py --stage b
  DEP=$LAST_ID
done
B_LAST=$DEP

# cifar10 only: corrupted test cells (independent of training)
CELLS_DEP="$B_LAST"
if [[ "$DATASET" == "cifar10" ]]; then
  CELLS_ID=$(sbatch --parsable scripts/slurm/lfcbm_genericity/cifar_embed_cells.sbatch)
  echo "${CELLS_ID} lfcbm-c10-cells" | tee -a "$JOBS_FILE"
  CELLS_DEP="${B_LAST}:${CELLS_ID}"
fi

# Propagation (resumable; give it two waves for the 50-cell I/O)
sub "lfcbm-${DATASET}-prop1" "$CELLS_DEP" "$SB" "$DATASET" run_propagation.py
sub "lfcbm-${DATASET}-prop2" "$LAST_ID" "$SB" "$DATASET" run_propagation.py

# d_k then capacity-matched ablation waves
sub "lfcbm-${DATASET}-dk" "$CELLS_DEP" "$SB" "$DATASET" run_concept_shift.py
DEP=$LAST_ID
for w in $(seq 1 "$ABL_WAVES"); do
  sub "lfcbm-${DATASET}-abl${w}" "$DEP" "$SB" "$DATASET" run_ablation.py
  DEP=$LAST_ID
done

echo "chain submitted for ${DATASET}; ids in ${JOBS_FILE}"
