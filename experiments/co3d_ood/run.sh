#!/usr/bin/env bash
# CO3D out-of-distribution viewpoints.
#   usage: experiments/co3d_ood/run.sh [hydrants|vases] [gpu]
# Stage 1 runs the search for each test object; stage 2 meshes the result and scores
# Chamfer + novel views against the CO3D pseudo-ground-truth.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && source .env
CATEGORY=${1:-hydrants}; GPU=${2:-0}
RUNS=${INFER3D_RUNS_ROOT:-$(pwd)/runs}
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled PYTHONPATH=$(pwd)

case "$CATEGORY" in
  hydrants) CSV=assets/co3d_test_paths_1080.csv ;;
  vases)    CSV=assets/co3d_vases_test_paths_1080.csv ;;
  *) echo "usage: $0 [hydrants|vases] [gpu]" >&2; exit 1 ;;
esac
LIFTER=$(python -c "from infer3d import config as c; print(c.LIFTER_CKPTS['$CATEGORY'])")

python experiments/co3d_ood/optimize.py \
    abs=diffae_abs +dataset=$CATEGORY general.run_indist=false \
    general.prefix=$CATEGORY-ood \
    opt.pretrained_ckpt="$LIFTER" \
    +general.test_imgs_csv_path="$(pwd)/$CSV"

python experiments/co3d_ood/run_eval_pipeline.py \
    --checkpoint_dir "$RUNS/$CATEGORY-ood" \
    --output_base_dir "$RUNS/$CATEGORY-ood-eval" \
    --dataset_name "$CATEGORY" --pretrained_ckpt "$LIFTER"

echo "Per-object metrics: $RUNS/$CATEGORY-ood-eval/*/results/{ours,baseline_ood}_with_icp.csv"
