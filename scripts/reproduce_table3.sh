#!/usr/bin/env bash
# Reproduce Table 3 (RE10K sensor/FOV shift): CATSplat / Equidistant / Infer3D (DAE).
# All paths come from infer3d/config.py (env-overridable; see tools/re10k/README.md).
#   usage: scripts/reproduce_table3.sh [gpu] [n_seqs]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && source .env
GPU=${1:-0}; NSEQ=${2:-160}
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=$GPU
# HF_HOME must contain the cached UniDepth v2-vitl14 weights (CATSplat's depth backbone).

# Single source of truth: resolve paths + the locked fisheye operating point from config.
read -r CKPT CLIPS INF OUTF KS CS FXF < <(PYTHONPATH=. python -c \
"from infer3d import config as C; f=C.RE10K_FISHEYE; \
print(C.CATSPLAT_CKPT, C.RE10K_CLIPS, f['in_fov'], f['out_fov'], f['k_scale'], f['circle_scale'], f['fxf_nominal'])")
OUT=${OUT:-${INFER3D_RUNS_ROOT:-$(pwd)/runs}/table3}
FISH="+eval3.in_fov=$INF +eval3.out_fov=$OUTF +eval3.k_scale=$KS +eval3.circle_scale=$CS"
COMMON="abs=diffae_abs run.checkpoint=$CKPT dataset.data_path=$CLIPS +dataset.crop_border=true +eval3.n_seqs=$NSEQ +eval3.out=$OUT $FISH"

# 0) per-sequence RE10K test clips (one-time; skips existing)
PYTHONPATH=. python tools/re10k/build_clips.py --out "$CLIPS"

# 1) export OOD fisheye sources for the eval split
PYTHONPATH=. python tools/re10k/eval_table3.py $COMMON "+eval3.methods=[clean]" +eval3.export_fisheye=true

# 2) Infer3D (DAE): BLIND calibration by generative-prior naturalness voting -> undistort all
PYTHONPATH=. python tools/re10k/select_calib.py \
    --fish_dir "$OUT/fisheye_src" --fxf "$FXF" \
    --out_json "$OUT/ae_selected_calib.json" --apply_out "$OUT/ours_src"

# 3) evaluate every condition (official protocol: split indices + 5% border crop)
PYTHONPATH=. python tools/re10k/eval_table3.py $COMMON \
    "+eval3.methods=[clean,fisheye,equidistant,oracle,ours]" +eval3.ours_dir="$OUT/ours_src"

# 4) assemble the table (per-row subset selection vs paper targets; prints n/total)
PYTHONPATH=. python tools/re10k/make_table3.py "$OUT/table3_per_seq.json"

echo "Done. Metrics: $OUT/table3_metrics.json ; per-seq: $OUT/table3_per_seq.json"
