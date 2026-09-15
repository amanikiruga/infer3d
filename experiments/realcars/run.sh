#!/usr/bin/env bash
# RealCars: synthetic-to-real appearance shift, Chamfer against per-scene pseudo-GT.
#   usage: experiments/realcars/run.sh [gpu] [first_scene] [last_scene]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && source .env
GPU=${1:-0}; FIRST=${2:-0}; LAST=${3:-19}
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled PYTHONPATH=$(pwd)
RUNS=${INFER3D_RUNS_ROOT:-$(pwd)/runs}
LIFTER=$(python -c "from infer3d import config as c; print(c.LIFTER_CKPTS['cars'])")
SCENES=$(python -c "from infer3d import config as c; print(c.REALCARS_ROOT)")

# 1. search, one scene at a time (assets/realcars_test_paths.csv fixes the scene list)
for i in $(seq -f "%02g" "$FIRST" "$LAST"); do
  IMG=$SCENES/$(sed -n "$((10#$i + 1))p" assets/realcars_test_paths.csv)
  [ -d "$RUNS/realcars_opt/$i" ] && { echo "skip $i (already done)"; continue; }
  python -u experiments/realcars/optimize.py \
      abs=diffae_abs +dataset=cars general.split=0 general.total_splits=1 \
      general.prefix=realcars_opt opt.pretrained_ckpt="$LIFTER" \
      +real_ood.image_path="$IMG" +real_ood.out_subdir="$i" \
      +real_ood.num_iterations=783 +real_ood.save_every=50 \
      +real_ood.narrow_s1_end=122 +real_ood.narrow_s2_end=302 \
      +real_ood.lambda_rgb=1.0 +real_ood.lambda_bg=0.5 +real_ood.lambda_depth=0.5 \
      +real_ood.lambda_dino=0.5 +real_ood.lambda_lpips=0.5 \
      +real_ood.w_reg=0.025 +real_ood.rank_w_reg_mult=1.0
done

# 2. optimized splats -> PLY, 3. ICP-aligned Chamfer vs pseudo-GT, 4. table
python experiments/realcars/splats_to_ply_realcars.py
python experiments/realcars/eval_chamfer_direct.py
python experiments/realcars/score.py
