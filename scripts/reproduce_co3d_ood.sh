#!/usr/bin/env bash
# CO3D OOD reproduction (Table 1). Stage A (test-time optimization) + Stage B
# (gs2mesh + ICP -> Chamfer + NVS). GPU required.
#   usage: scripts/reproduce_co3d_ood.sh {hydrants|vases} [gpu]
# Budget ~55 min of H100 time per object in Stage A, plus Stage B on top.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && source .env
DSET=${1:-hydrants}; GPU=${2:-0}
export WANDB_MODE=disabled
ASSETS=$(pwd)/assets
RUNS=${INFER3D_RUNS_ROOT:-$(pwd)/runs}
case "$DSET" in
  hydrants) LIFTER=$SPLATTER_REPO_ROOT/experiments_out/2025-10-07/16-13-13/model_latest.pth
            CSV=$ASSETS/co3d_test_paths_1080.csv ;;
  vases)    LIFTER=$SPLATTER_REPO_ROOT/experiments_out/2026-01-15/14-55-34/model_latest.pth
            CSV=$ASSETS/co3d_vases_test_paths_1080.csv ;;
  *) echo "unknown dataset $DSET"; exit 1 ;;
esac
# general.prefix resolves relative to $INFER3D_RUNS_ROOT, so it must not repeat "runs/".
PREFIX=$DSET-ood

# Stage A: test-time optimization (DiffAE prior), OOD SE(3) pose
PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python tools/optimize_co3d_diffae.py \
    abs=diffae_abs +dataset=$DSET general.run_indist=false \
    general.prefix="$PREFIX" \
    opt.pretrained_ckpt="$LIFTER" \
    +general.test_imgs_csv_path="$CSV"

# Stage B: Chamfer + novel-view metrics via gs2mesh + Sim(3)-ICP
PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python tools/eval/run_eval_pipeline.py \
    --checkpoint_dir "$RUNS/$PREFIX" \
    --output_base_dir "$RUNS/eval" \
    --dataset_name "$DSET" --pretrained_ckpt "$LIFTER"

echo "Done. Score with: RESULTS_ROOT=$RUNS/eval python tools/make_tables.py"
