#!/bin/bash
# ImageNet corruption-propagation FULL re-dump — all 50 cells (10 families ×
# 5 severities) — with --dump-stage-tensors, tar-staged to $TMPDIR.
#
# One coherent, reproducible realization of the whole suite:
#   * §3.3 mCE/RmCE consumes all 50 cells' summaries.
#   * §3.4 concept robustness (experiments/concept_retention/) consumes the
#     10 sev-3 cells' full activations (retention / intrusion / phi-drift).
# Replaces the prior mixed state (45 argmax-only + 5 sev-5 full dumps).
#
# DETERMINISM: PYTHONHASHSEED is fixed at launch (below). The 6 sensor families
# seed their per-image masks from hash((image_id, corruption, severity)); that
# string hash is salted per-process unless PYTHONHASHSEED is set BEFORE the
# interpreter starts. We pin it here; seeding.py honours a pre-set value (and
# re-execs to pin it itself when unset), so this export keeps the whole suite
# — including any resubmit of a failed cell — on one salt, giving a
# reproducible sensor realization. The 4 Hendrycks families load pre-generated
# ImageNet-C from disk → already byte-identical regardless.
#
# Tar staging (cp single tar -> $TMPDIR local SSD -> extract) avoids the Lustre
# metadata storm of reading 50k loose JPEGs/cell. Mirrors the per-cell staging
# in imagenet_rn50_c_propagation/_common.sh. Routing is a pure function of the
# cell index (corruptions[task//5]); the staging fn handles all 0-49:
#   task 0-19  Hendrycks (motion_blur,gaussian_noise,jpeg_compression,fog)
#              -> pre-generated ImageNet-C family tar -> DCBM_IMGNET_C_DIR
#   task 20-49 sensor (6 families) -> on-the-fly from clean test tar -> DCBM_IMAGENET_DIR
# The deployed bank stays on Lustre; only the IMAGE roots are redirected.
#
# Each task is an independent <28 min job (~3.5-6 min CLIP forward + ~90 s setup
# + ~0.5-2 min staging); SLURM runs the 50 as separate short-slot jobs.
#
# Submit:  sbatch scripts/slurm/imagenet_c_propagation/05_dump_sev3_staged.sbatch
# Writes:  experiments/imagenet_c_propagation/<corr>_s<sev>/per_image/*.pt (~500 MB/cell, ~25 GB total)
# Prereqs on $WS: imagenet_test.tar + raw/ImageNet-C/_tars/imagenet_c_{motion_blur,
#                 gaussian_noise,jpeg_compression,fog}.tar  (all present 2026-06-24)

#SBATCH --job-name=in1k-c-redump-all
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0:28:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet_c_propagation/_common.sh"

# Fix string-hash salt at launch so sensor per-image masks are reproducible.
export PYTHONHASHSEED=0

# --- per-cell tar staging (mirrors imagenet_rn50_c_propagation/_common.sh) ---
_HENDRYCKS=(motion_blur gaussian_noise jpeg_compression fog)  # families 0..3

stage_corruption_inputs() {
    local tid="$1"
    local fam_idx=$(( tid / 5 ))
    if [ "$fam_idx" -lt 4 ]; then
        local corr="${_HENDRYCKS[$fam_idx]}"
        local tar="${WS}/raw/ImageNet-C/_tars/imagenet_c_${corr}.tar"
        echo "[stage] Hendrycks cell tid=${tid} family=${corr}"
        [ -f "$tar" ] || { echo "ERROR: missing corruption tar $tar"; exit 1; }
        mkdir -p "$TMPDIR/ImageNet-C"
        local t0=$(date +%s)
        cp "$tar" "$TMPDIR/"
        tar xf "$TMPDIR/imagenet_c_${corr}.tar" -C "$TMPDIR/ImageNet-C"
        rm "$TMPDIR/imagenet_c_${corr}.tar"
        export DCBM_IMGNET_C_DIR="$TMPDIR/ImageNet-C"
        echo "[stage] staged in $(( $(date +%s) - t0 ))s -> DCBM_IMGNET_C_DIR=${DCBM_IMGNET_C_DIR}"
    else
        local tar="${WS}/imagenet_test.tar"
        echo "[stage] Sensor cell tid=${tid} -> clean test tree"
        [ -f "$tar" ] || { echo "ERROR: missing test tar $tar (run 00_make_test_tar.sbatch)"; exit 1; }
        mkdir -p "$TMPDIR/imagenet"
        local t0=$(date +%s)
        cp "$tar" "$TMPDIR/"
        tar xf "$TMPDIR/imagenet_test.tar" -C "$TMPDIR/imagenet"
        rm "$TMPDIR/imagenet_test.tar"
        export DCBM_IMAGENET_DIR="$TMPDIR/imagenet"
        echo "[stage] staged in $(( $(date +%s) - t0 ))s -> DCBM_IMAGENET_DIR=${DCBM_IMAGENET_DIR}"
    fi
}

stage_corruption_inputs "${SLURM_ARRAY_TASK_ID}"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/imagenet_c_propagation"
srun python "${RUN_DIR}/run_cell.py" --task-id "${SLURM_ARRAY_TASK_ID}" --dump-stage-tensors

echo "------------------------------------------------------------"
echo "Finished: $(date -Iseconds)"
