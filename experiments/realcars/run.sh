#!/usr/bin/env bash
# RealCars: synthetic-to-real appearance shift, Chamfer against ARKit pseudo-ground-truth.
#   usage: experiments/realcars/run.sh [gpu]
# Assumes the per-scene optimization outputs are already under $INFER3D_RUNS_ROOT/realcars_opt.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && source .env
export CUDA_VISIBLE_DEVICES=${1:-0} WANDB_MODE=disabled PYTHONPATH=$(pwd)
RUNS=${INFER3D_RUNS_ROOT:-$(pwd)/runs}

python experiments/realcars/splats_to_ply_realcars.py      # optimized splats -> PLY
python experiments/realcars/eval_chamfer_direct.py         # ICP-aligned Chamfer vs pseudo-GT
python experiments/realcars/build_table_4col.py            # assemble the table

echo "Results: $RUNS/realcars/"
