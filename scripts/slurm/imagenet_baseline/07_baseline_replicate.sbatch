#!/bin/bash
# Step 7: Replicate the best-val sweep cell across 3 seeds. Reads every
# sweep_shard_*.jsonl, picks argmax-val cell, runs seeds 0..2 back-to-back
# and writes summary.json with pass/fail verdict vs target in config.
#
# 3 seeds not 5 (KMeans seed hardcoded
# upstream, so "N seeds" only varies linear-probe init; CIFAR-10 4-seed
# std was 0.0004 microscopic).
#
# If cap is hit: seeds.jsonl is appended line-by-line as each seed
# finishes. --phase summarize will read whatever seeds.jsonl has when
# the cap hits; a truncated final line is skipped (see the CIFAR-10 Apr
# 20 recovery). Resubmit with a longer cap or rerun missing seeds
# separately.
#
# Submit chained after the sweep array:
#   sbatch --dependency=afterok:<sweeparrayid> \
#       scripts/slurm/imagenet1000/07_baseline_replicate.sbatch
#
# Reads:  experiments/imagenet_baseline/sweep_shard_*.jsonl
# Writes:
#   experiments/imagenet_baseline/seeds.jsonl
#   experiments/imagenet_baseline/summary.json
#   experiments/imagenet_baseline/git_commit.txt

#SBATCH --job-name=in1k-replicate
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/imagenet1000/_common.sh"

RUN_DIR="${SLURM_SUBMIT_DIR}/experiments/imagenet_baseline"

start_heartbeat 600 "${RUN_DIR}"

srun python "${RUN_DIR}/run.py" --phase replicate

stop_heartbeat

echo "------------------------------------------------------------"
if [ -f "${RUN_DIR}/summary.json" ]; then
    echo "summary.json: $(stat -c %s "${RUN_DIR}/summary.json") bytes"
    python -c "
import json
with open('${RUN_DIR}/summary.json') as f:
    s = json.load(f)
rep = s['metrics']['replication']
print(f'  mean_test: {rep[\"mean_test_accuracy\"]:.4f} +/- {rep[\"std_test_accuracy\"]:.4f}')
print(f'  target:    {s[\"target_mean_test_accuracy\"]}')
print(f'  passed:    {s[\"passed\"]}')
"
else
    echo "summary.json: MISSING"
fi
echo "Finished: $(date -Iseconds)"
