#!/bin/bash
# Leave-one-corruption-out (LOCO) test of d_k selection, ImageNet, all three CBM families.
#
# For each held-out corruption c: rank concepts by d_k over the other 9 corruptions (all
# severities) on the disjoint 50k train pool (experiments/imagenet_dk_disjoint_pool/), prune
# to the family's best K, refit the head on the 128k refit pool (3 seeds), and read accuracy
# on the held-out c (sev 3, 50k eval). Arms: dk_loco_<c> x10, dk_disjoint (all 10 seen),
# rand0-2. Results land in experiments/imagenet_dk_disjoint_pool/loco/<family>/, never in
# the main ablation files. Resumable (results.jsonl), so waves are chained afterany.
#
#   bash scripts/slurm/dk_disjoint_loco/submit.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=$PWD/experiments/imagenet_dk_disjoint_pool
FAMS=(fog gaussian_noise motion_blur patch_masking salt_and_pepper jpeg_compression
      row_column_dropout random_pixel_dropout sparse_random_occlusion structured_pixel_dropout)
LOCO=$(printf "dk_loco_%s," "${FAMS[@]}")
RANKS="${LOCO}dk_disjoint,rand0,rand1,rand2"

# --- LF-CBM / PCBM-mmL: GPU LBFGS refit, 42 fits each at K=800, 30-min waves ---
submit_text() {  # <name> <grid npz> <table csv> <out subdir> <runner args...>
    local name=$1 grid=$2 table=$3 out=$4; shift 4
    local p="" ids=""
    for w in 1 2 3 4 5; do
        p=$(DK_LOCO_GRID=$grid DK_DISJOINT_TABLE=$table ABL_RANKS=$RANKS ABL_SIZES=800 \
            ABL_OUT_DIR=$D/loco/$out \
            sbatch --parsable --job-name=$name ${p:+--dependency=afterany:$p} \
            scripts/slurm/genericity_full_val_ablation.sbatch "$@")
        ids="$ids $p"
    done
    echo "$name:$ids"
}
LF=$(submit_text loco-lf $D/lfcbm_shift_grids.npz $D/lfcbm_concept_shift_table.csv lfcbm \
     experiments/imagenet_lfcbm_genericity_full_val/run_ablation.py)
PC=$(submit_text loco-pc $D/pcbm_mmL_shift_grids.npz $D/pcbm_mmL_concept_shift_table.csv pcbm_mmL \
     experiments/imagenet_pcbm_genericity_full_val/run_ablation.py --bank mmL)

# --- DCBM: sklearn refit on CPU (~30 min/fit), one job per arm (3 seeds) at K=400 ---
DC="loco-dcbm:"
for rk in ${RANKS//,/ }; do
    j=$(DK_LOCO_GRID=$D/dcbm_shift_grids.npz DK_DISJOINT_TABLE=$D/dcbm_concept_shift_table.csv \
        RANK_BY=$rk SIZES=400 INCLUDE_NATIVE=0 FIT_N=0 MAX_ITER=1000 \
        OUT_DIR=$D/loco/dcbm/$rk \
        NATIVE_CACHE=$PWD/experiments/imagenet_concept_shift_robust_set/capacity_matched \
        sbatch --parsable --job-name=loco-dcbm --cpus-per-task=32 --mem=128G --time=4:00:00 \
        scripts/slurm/cpu_task.sbatch \
        python experiments/imagenet_concept_shift_robust_set/ablation_capacity_matched.py)
    DC="$DC $j"
done

printf "%s  LOCO: %s | %s | %s\n" "$(date -Iseconds)" "$LF" "$PC" "$DC" | tee -a $D/slurm_jobs.txt
