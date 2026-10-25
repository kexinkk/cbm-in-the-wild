#!/bin/bash
# Concept selection with d_k estimated on the disjoint train pool, and leave-one-corruption-out
# (LOCO): for each held-out corruption c, rank concepts by d_k over the other 9 corruptions,
# keep the family's best K, refit the head on the 128k held-out pool (seeds 0-2) and read
# accuracy on c (sev 3, 50k test). Arms: dk_loco_<c> x10, dk_disjoint (all 10 seen), rand0-2.
# Needs compute_dk.py outputs. Resumable: finished (arm, K, seed) cells are skipped.
#
#   bash experiments/imagenet_dk_disjoint_pool/run_loco.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
D=experiments/imagenet_dk_disjoint_pool
FAMS=(fog gaussian_noise motion_blur patch_masking salt_and_pepper jpeg_compression
      row_column_dropout random_pixel_dropout sparse_random_occlusion structured_pixel_dropout)
RANKS="$(printf "dk_loco_%s," "${FAMS[@]}")dk_disjoint,rand0,rand1,rand2"

# LF-CBM and PCBM-mmL at K=800 (GPU head refit)
DK_LOCO_GRID=$D/lfcbm_shift_grids.npz DK_DISJOINT_TABLE=$D/lfcbm_concept_shift_table.csv \
ABL_RANKS=$RANKS ABL_SIZES=800 ABL_OUT_DIR=$D/loco/lfcbm \
    python experiments/imagenet_lfcbm_genericity/run_ablation.py
DK_LOCO_GRID=$D/pcbm_mmL_shift_grids.npz DK_DISJOINT_TABLE=$D/pcbm_mmL_concept_shift_table.csv \
ABL_RANKS=$RANKS ABL_SIZES=800 ABL_OUT_DIR=$D/loco/pcbm_mmL \
    python experiments/imagenet_pcbm_genericity/run_ablation.py --bank mmL

# DCBM at K=400 (CPU head refit, ~30 min per fit), one output dir per arm
for rk in ${RANKS//,/ }; do
    DK_LOCO_GRID=$D/dcbm_shift_grids.npz DK_DISJOINT_TABLE=$D/dcbm_concept_shift_table.csv \
    RANK_BY=$rk SIZES=400 INCLUDE_NATIVE=0 OUT_DIR=$D/loco/dcbm/$rk \
    NATIVE_CACHE=experiments/imagenet_concept_shift_robust_set/capacity_matched \
        python experiments/imagenet_concept_shift_robust_set/ablation_capacity_matched.py
done
python $D/aggregate_loco.py
