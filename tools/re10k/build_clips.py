#!/usr/bin/env python3
"""
Build the per-sequence RE10K test clips ({seq}.mp4) the CATSplat dataloader expects.

The loader reads frame index i from {data_path}/{seq}.mp4, where i indexes the
pose/timestamp array. The raw RealEstate10K release ships full source videos + per-seq
pose/timestamp .txt files but NOT the per-seq clips; this extracts the frame at each
timestamp[i] from the source video, in order, so clip-frame i == pose-frame i.

Raw-data paths (env-overridable):
  RE10K_PICKLE     flash3d annotation pickle (seq -> {timestamps,...})
  RE10K_POSE_DIR   dir of {seq}.txt pose files (first line -> youtube id)
  RE10K_VIDEO_DIR  dir of {youtube_id}.mp4 source videos
Output dir (--out) defaults to infer3d.config.RE10K_CLIPS.

  PYTHONPATH=. python tools/re10k/build_clips.py --out $RE10K_CLIPS
"""
import os, sys, gzip, pickle, argparse
from pathlib import Path
import cv2

sys.path.insert(0, os.getcwd())
try:
    from infer3d import config as C
    _DEFAULT_OUT = C.RE10K_CLIPS
    _EXTERN = C.EXTERN_ROOT
except Exception:
    _DEFAULT_OUT = None
    _EXTERN = os.getenv("INFER3D_EXTERN_ROOT", ".")

_RAW = os.getenv("RE10K_RAW_ROOT", f"{_EXTERN}/datasets/re10k")
PICKLE    = os.getenv("RE10K_PICKLE",    f"{_RAW}/flash3d_anns/catsplat_test_256.pickle.gz")
POSE_DIR  = os.getenv("RE10K_POSE_DIR",  f"{_RAW}/RealEstate10K/test")
VIDEO_DIR = os.getenv("RE10K_VIDEO_DIR", f"{_RAW}/videos")


def youtube_id(seq):
    txt = Path(POSE_DIR) / f'{seq}.txt'
    if not txt.exists():
        return None
    line = open(txt).readline().strip()
    return line.split('v=')[-1].split('&')[0] if 'v=' in line else line.rsplit('/', 1)[-1]


def build_clip(seq, timestamps, out_path, size=256):
    yid = youtube_id(seq)
    if yid is None:
        return 'no pose file'
    vid = Path(VIDEO_DIR) / f'{yid}.mp4'
    if not vid.exists():
        return f'no video {yid}'
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return f'cannot open {yid}'
    fps = cap.get(cv2.CAP_PROP_FPS); ntot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        cap.release(); return 'fps=0'
    frames = []
    for ts in timestamps:
        fn = max(0, min(ntot - 1, int((ts / 1_000_000.0) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, fr = cap.read()
        if not ret:
            cap.release(); return f'read fail @frame {fn}'
        frames.append(fr)
    cap.release()
    if not frames:
        return 'no frames'
    h, w = frames[0].shape[:2]
    W = int(round(w * size / h))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, size))
    for fr in frames:
        vw.write(cv2.resize(fr, (W, size)))
    vw.release()
    return f'OK ({len(frames)} frames, {W}x{size})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=_DEFAULT_OUT, required=_DEFAULT_OUT is None)
    ap.add_argument('--seqs', nargs='*', default=None, help='specific seq ids; default = all in pickle')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with gzip.open(PICKLE, 'rb') as f:
        data = pickle.load(f)
    seqs = args.seqs if args.seqs else sorted(data.keys())
    if args.limit:
        seqs = seqs[:args.limit]
    i, N = map(int, args.shard.split('/')); seqs = seqs[i::N]
    ok = 0
    for s in seqs:
        out = Path(args.out) / f'{s}.mp4'
        if out.exists() and out.stat().st_size > 1000:
            ok += 1; continue
        if s not in data:
            print(f'{s}: not in pickle', flush=True); continue
        msg = build_clip(s, data[s]['timestamps'], out)
        ok += msg.startswith('OK')
        print(f'{s}: {msg}', flush=True)
    print(f'\nDONE shard {args.shard}: {ok}/{len(seqs)} clips ready', flush=True)


if __name__ == '__main__':
    main()
