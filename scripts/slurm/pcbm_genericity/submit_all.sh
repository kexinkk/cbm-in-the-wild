#!/bin/bash
# Submit the PCBM genericity chain on the 30-min short GPU partitions,
# afterany-chained (timeouts harmless: every stage is skip-if-exists /
# resumable). Requires Stage A (bank build) COMPLETED first — model_cav/,
# model_mm/ with cavs/proj_mean/proj_std present (stage_a_bank.sbatch).
#
#   bash scripts/slurm/pcbm_genericity/submit_all.sh              # both banks
#   bash scripts/slurm/pcbm_genericity/submit_all.sh cav          # one bank
#
# Per bank:
#   stage-B (faithful elastic-net head) ─→ d_k ─→ ablation waves
#                                        └─→ propagation
# d_k's w_imp column and propagation both read the deployed head, so both wait
# on stage-B; the ablation refit head is independent (plain L2, see ablation.py).

set -euo pipefail
cd "$(dirname "$0")/../../.."

if [[ $# -gt 0 ]]; then BANKS=("$@"); else BANKS=(cav mm); fi

SB=scripts/slurm/pcbm_genericity/stage.sbatch
JOBS_FILE="experiments/imagenet_pcbm_genericity/slurm_jobs.txt"
B_WAVES=2
ABL_WAVES=2

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

for bank in "${BANKS[@]}"; do
  # stage B: faithful PCBM reference head (resumable waves)
  sub "pcbm-${bank}-B1" "" "$SB" train_pcbm.py --stage b --bank "$bank"
  DEP=$LAST_ID
  for w in $(seq 2 "$B_WAVES"); do
    sub "pcbm-${bank}-B${w}" "$DEP" "$SB" train_pcbm.py --stage b --bank "$bank"
    DEP=$LAST_ID
  done
  B_LAST=$DEP

  # d_k (w_imp needs the deployed head)
  sub "pcbm-${bank}-dk" "$B_LAST" "$SB" run_concept_shift.py --bank "$bank"
  DEP=$LAST_ID
  for w in $(seq 1 "$ABL_WAVES"); do
    sub "pcbm-${bank}-abl${w}" "$DEP" "$SB" run_ablation.py --bank "$bank"
    DEP=$LAST_ID
  done

  # propagation (needs the deployed head)
  sub "pcbm-${bank}-prop" "$B_LAST" "$SB" run_propagation.py --bank "$bank"
done

echo "chain submitted for banks: ${BANKS[*]}; ids in ${JOBS_FILE}"
