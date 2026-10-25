#!/bin/bash
# Download the released checkpoints from the GitHub release and unpack them in place.
#
#   bash scripts/download_checkpoints.sh            # ImageNet + CIFAR-10
#   bash scripts/download_checkpoints.sh imagenet   # one dataset
#
# checkpoints_imagenet.tar.gz  DCBM (K=2048) data/checkpoints/imagenet_baseline/,
#                              LF-CBM experiments/imagenet_lfcbm_genericity/model/,
#                              PCBM-mmL experiments/imagenet_pcbm_genericity/model_mmL/
# checkpoints_cifar10.tar.gz   the same three families on CIFAR-10 (PCBM: cav, mm, mmL)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${TAG:-v1.0}"
URL="https://github.com/kexinkk/cbm-in-the-wild/releases/download/${TAG}"
declare -A SHA=(
  [imagenet]=19aa20849c4b90fcb09db9e2007ee9ed7ba0d8b6021f1cee30e3b94dae8006db
  [cifar10]=955eec0baa3904c9176c75a9d9e0de5148d6ccec1e8f1f1724401c71422dd10b
)
for ds in "${@:-imagenet cifar10}"; do
  for d in $ds; do
    f="checkpoints_${d}.tar.gz"
    curl -fL -o "$f" "${URL}/${f}"
    echo "${SHA[$d]}  $f" | sha256sum -c -
    tar xzf "$f" && rm "$f"
  done
done
