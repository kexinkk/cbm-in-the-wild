#!/bin/bash
# Submit the full-val capacity-matched refit as WAVES chained afterany, so a 24h TIMEOUT
# resumes for free from the per-cell cache (native_K*.pt reused + results.jsonl). Finished
# cells exit later waves in seconds (skip-if-cached).
#
#   bash scripts/slurm/concept_shift_robust_set_full_val/submit.sh
#   WAVES=3 bash scripts/slurm/concept_shift_robust_set_full_val/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

WAVES="${WAVES:-2}"
SB=scripts/slurm/concept_shift_robust_set_full_val/run.sbatch
dep=""
for w in $(seq 1 "${WAVES}"); do
  if [ -z "${dep}" ]; then
    jid=$(sbatch --parsable "${SB}")
  else
    jid=$(sbatch --parsable --dependency="afterany:${dep}" "${SB}")
  fi
  echo "wave ${w}/${WAVES} -> ${jid}"
  dep="${jid}"
done
echo
echo "Artifacts land in experiments/imagenet_concept_shift_robust_set_full_val/"
echo "If still unfinished after ${WAVES} waves: re-run this script (resumes from cache)."
