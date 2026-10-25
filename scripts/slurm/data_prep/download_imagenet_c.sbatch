#!/bin/bash
# Download + selectively extract ImageNet-C for the 4 selected
# corruption types (motion_blur, gaussian_noise, jpeg_compression, fog).
#
# Source: Zenodo record 2235448 (Hendrycks). 4 category tars (~76 GB combined),
# selective extract → ~25 GB on disk. Tars removed after extraction (default).
#
# CPU-only job; cpu_il dispatches in minutes. Generous --time margin since
# wall-clock is dominated by network I/O (~13 min @ 100 MB/s) + selective
# tar streaming (~30-60 min depending on filesystem).
#
# Submit:
#   sbatch scripts/slurm/data_prep/download_imagenet_c.sbatch
#
# Restartable: skip-if-exists per corruption (presence of <corruption>/5/).

#SBATCH --job-name=imagenet-c-dl
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/data_prep/_common.sh"

srun python -m scripts.data_prep.imagenet_c_download

echo "------------------------------------------------------------"
TARGET_ROOT="${SLURM_SUBMIT_DIR}/data/raw/ImageNet-C"
if [ -d "${TARGET_ROOT}" ]; then
    echo "ImageNet-C tree under ${TARGET_ROOT}:"
    for c in motion_blur gaussian_noise jpeg_compression fog; do
        if [ -d "${TARGET_ROOT}/${c}/5" ]; then
            n=$(find "${TARGET_ROOT}/${c}/5" -name '*.JPEG' | wc -l)
            echo "  ${c}/5: ${n} JPEGs across $(ls "${TARGET_ROOT}/${c}/5" | wc -l) wnids"
        else
            echo "  ${c}: MISSING severity 5"
        fi
    done
fi
echo "Finished: $(date -Iseconds)"
