#!/usr/bin/env bash
# ShapeNet-NMR OOD reproduction (Table 2). Stage A only (image metrics; the NMR
# pipeline computes no Chamfer). GPU required. Splits across GPUs via split/total.
#   usage: scripts/reproduce_shapenet_ood.sh {se3|so3} [split] [total_splits] [gpu]
# NOTE: general.prefix resolves relative to $INFER3D_RUNS_ROOT (default ./runs).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && source .env
VIEW=${1:-se3}; SPLIT=${2:-0}; TOTAL=${3:-1}; GPU=${4:-0}
export WANDB_MODE=disabled
ASSETS=$(pwd)/assets
LIFTER=$SPLATTER_REPO_ROOT/experiments_out/2025-08-06/12-11-04/model_latest.pth

if [ "$VIEW" = "se3" ]; then SCRIPT=tools/optimize_nmr_se3.py; else SCRIPT=tools/optimize_nmr_so3.py; fi

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
    +dataset=shapenet-nmr general.split=$SPLIT general.total_splits=$TOTAL \
    general.prefix="shapenet-nmr-$VIEW" \
    opt.pretrained_ckpt="$LIFTER" \
    general.data_example_ids_path="$ASSETS/nmr_test_split.json"

echo "Done. Score with: PYTHONPATH=. RESULTS_ROOT=${INFER3D_RUNS_ROOT:-$(pwd)/runs} python tools/make_tables.py"
