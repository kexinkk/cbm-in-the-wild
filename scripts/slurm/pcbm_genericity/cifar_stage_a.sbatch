#!/bin/bash
# PCBM Stage A for CIFAR-10: build cav concept bank + standardizer + L2 sanity
# head. Stages the Broden concept zip to $TMPDIR (single sequential read →
# extract on local SSD, no Lustre metadata storm over the 33k loose pngs) and
# exports PCBM_BRODEN_DIR. Skip-if-exists per bank → afterany-safe.
#
#   sbatch scripts/slurm/pcbm_genericity/cifar_stage_a.sbatch

#SBATCH --job-name=pcbm-c10-A
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0:25:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/pcbm_genericity/_common.sh"

BRODEN_ZIP="${WS}/pcbm/broden_concepts.zip"
echo "Staging ${BRODEN_ZIP} → ${TMPDIR}"
cp "${BRODEN_ZIP}" "${TMPDIR}/broden_concepts.zip"
unzip -q "${TMPDIR}/broden_concepts.zip" -d "${TMPDIR}"
export PCBM_BRODEN_DIR="${TMPDIR}/broden_concepts"
echo "PCBM_BRODEN_DIR=${PCBM_BRODEN_DIR}  ($(ls "${PCBM_BRODEN_DIR}" | wc -l) concepts)"

srun python experiments/cifar10_pcbm_genericity/train_pcbm.py --stage a
