#!/usr/bin/env bash
# RealEstate10K under an unseen fisheye lens, with blind calibration.
#   usage: experiments/re10k_fisheye/run.sh [gpu] [n_seqs]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && source .env
GPU=${1:-0}; NSEQ=${2:-160}
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled PYTHONPATH=$(pwd)
RUNS=${INFER3D_RUNS_ROOT:-$(pwd)/runs}; OUT=$RUNS/re10k

read -r CKPT CLIPS INF OUTF KS CS FXF < <(python -c \
"from infer3d import config as C; f=C.RE10K_FISHEYE; \
print(C.CATSPLAT_CKPT, C.RE10K_CLIPS, f['in_fov'], f['out_fov'], f['k_scale'], f['circle_scale'], f['fxf_nominal'])")
FISH="+eval3.in_fov=$INF +eval3.out_fov=$OUTF +eval3.k_scale=$KS +eval3.circle_scale=$CS"
COMMON="abs=diffae_abs run.checkpoint=$CKPT dataset.data_path=$CLIPS +dataset.crop_border=true +eval3.n_seqs=$NSEQ +eval3.out=$OUT $FISH"

E=experiments/re10k_fisheye
python $E/build_clips.py --out "$CLIPS"                              # one-time; skips existing
python $E/eval_table3.py $COMMON "+eval3.methods=[clean]" +eval3.export_fisheye=true
python $E/select_calib.py --fish_dir "$OUT/fisheye_src" --fxf "$FXF" \
       --out_json "$OUT/ae_selected_calib.json" --apply_out "$OUT/ours_src"
python $E/eval_table3.py $COMMON "+eval3.methods=[clean,fisheye,equidistant,oracle,ours]" \
       +eval3.ours_dir="$OUT/ours_src"
python $E/make_table3.py "$OUT/table3_per_seq.json"

echo "Results: $OUT/table3_metrics.json"
