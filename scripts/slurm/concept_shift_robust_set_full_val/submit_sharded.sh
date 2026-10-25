#!/bin/bash
# Full-val capacity-matched grid, PARALLELIZED: one shard job per arm (native + 7 B rankings),
# each writing to its own shards/<tag>/ dir, then a fast merge (afterany) into the canonical
# CSV. Every (arm,K,seed) fit is independent, so 8 shards run concurrently -> ~7 h wall-clock
# instead of ~45-70 h serial. Native banks are shared read-only (NATIVE_CACHE), no rebuild.
#
#   bash scripts/slurm/concept_shift_robust_set_full_val/submit_sharded.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

FV="experiments/imagenet_concept_shift_robust_set_full_val"
SB="scripts/slurm/concept_shift_robust_set_full_val"

# tag | RANK_BY (empty = native only) | INCLUDE_NATIVE
SHARDS=(
  "native|:|1"
  "dk_hold3|dk_hold3|0"
  "rand0|rand0|0"
  "rand1|rand1|0"
  "rand2|rand2|0"
  "geom|geom|0"
  "dkres_geom|dkres_geom|0"
  "geom_ar|geom_ar|0"
)

JIDS=()
for spec in "${SHARDS[@]}"; do
  IFS='|' read -r tag rb inc <<< "$spec"
  [ "$rb" = ":" ] && rb=""                       # ':' sentinel -> empty RANK_BY (native shard)
  out="${PWD}/${FV}/shards/${tag}"
  mkdir -p "${out}/capacity_matched"
  jid=$(sbatch --parsable --job-name="cmfv-${tag}" \
        --export=ALL,OUT_DIR="${out}",RANK_BY="${rb}",INCLUDE_NATIVE="${inc}" \
        "${SB}/shard.sbatch")
  printf 'shard %-12s -> %s   (RANK_BY=%q native=%s)\n' "${tag}" "${jid}" "${rb}" "${inc}"
  JIDS+=("${jid}")
done

dep=$(IFS=:; echo "${JIDS[*]}")
mjid=$(sbatch --parsable --dependency="afterany:${dep}" "${SB}/merge.sbatch")
echo
echo "merge -> ${mjid}  (afterany:${dep})"
echo "8 shards + merge submitted. Canonical CSV lands at ${FV}/ablation_capacity_matched.csv"
echo "If a shard TIMEOUTs (24h): re-run this script — cached cells skip, only missing ones refit."
