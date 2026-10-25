#!/bin/bash
# Grid-parity extension (2026-09-17): add the mm + mmL PCBM banks to CIFAR-10 so
# CIFAR PCBM reaches parity with ImageNet {cav, mm, mmL}. cav already exists
# (model_cav/, faithful head clean 0.948) and is left untouched (stage-A
# skip-if-exists per bank).
#
#   bash scripts/slurm/pcbm_genericity/submit_cifar_mm_mmL.sh
#
# Chain (all 30-min short GPU slots, afterany-chained, resumable):
#   A  : stage-A (Broden bank build) — cav SKIPS, mm builds        [cifar_stage_a.sbatch]
#   aL : stage-aL — mmL text bank (CLIP-text of CIFAR LF-CBM 137)  [cifar_stage.sbatch]
#   B  : per bank {mm, mmL}: stage-B glm_saga faithful-head waves  [cifar_stage.sbatch]
#
# Stage A / aL write an L2 *sanity* head; stage B overwrites W_g/b_g with the
# faithful elastic-net head (the deployed model + the clean_test_top1 used to
# compare the CIFAR PCBM banks).

set -euo pipefail
cd "$(dirname "$0")/../../.."

SB_A=scripts/slurm/pcbm_genericity/cifar_stage_a.sbatch
SB=scripts/slurm/pcbm_genericity/cifar_stage.sbatch
JOBS_FILE="experiments/cifar10_pcbm_genericity/slurm_jobs.txt"
B_WAVES=2   # CIFAR banks are small (K<=166); 2 waves is ample warm-restart budget

echo "# mm+mmL parity chain submitted $(date -Iseconds)" >> "$JOBS_FILE"

# --- stage A: build the mm bank (cav skips) ---
A_ID=$(sbatch --parsable --job-name=pcbm-c10-A "$SB_A")
echo "${A_ID} pcbm-c10-A(cav-skip+mm)" | tee -a "$JOBS_FILE"

# --- stage aL: build the mmL text bank (no Broden; needs LF-CBM concepts.txt) ---
AL_ID=$(sbatch --parsable --job-name=pcbm-c10-aL "$SB" train_pcbm.py --stage aL)
echo "${AL_ID} pcbm-c10-aL(mmL)" | tee -a "$JOBS_FILE"

sub() {  # sub <name> <dep> <args...>
  local name="$1" dep="$2"; shift 2
  local id
  id=$(sbatch --parsable --job-name="$name" --dependency="afterany:${dep}" "$@")
  echo "${id} ${name}" | tee -a "$JOBS_FILE"
  LAST_ID="$id"
}

# --- stage B: faithful elastic-net head per new bank ---
# mm depends on stage A; mmL depends on stage aL.
declare -A DEP0=( [mm]="$A_ID" [mmL]="$AL_ID" )
for bank in mm mmL; do
  DEP="${DEP0[$bank]}"
  for w in $(seq 1 "$B_WAVES"); do
    sub "pcbm-c10-${bank}-B${w}" "$DEP" "$SB" train_pcbm.py --stage b --bank "$bank"
    DEP="$LAST_ID"
  done
done

echo "CIFAR PCBM mm+mmL parity chain submitted; ids in ${JOBS_FILE}"
