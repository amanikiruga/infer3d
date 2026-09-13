"""Score a Task-A baseline render dir vs bundle targets, + montage/orbit video.

Uses scorer.score_views (identical PSNR/SSIM/LPIPS to our paper pipeline).
Also makes a sanity montage: input | reproj | [render|target] pairs, and an
orbit mp4 (render-over-target) so failures are visible, not just numeric.

  python score_task_a.py --method nfi --render_sub renders_native30 [--only OBJ]
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse
import json
import os
import sys

import numpy as np
import torch

WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
from scorer import score_views, bundle_targets, load_png  # noqa


def montage(paths, cols, path):
    from PIL import Image
    ims = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    h, w = ims[0].shape[:2]
    rows = (len(ims) + cols - 1) // cols
    canvas = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im
    Image.fromarray(canvas).save(path)


def orbit_video(render_paths, target_paths, path, fps=8):
    import imageio.v2 as imageio
    frames = []
    for rp, tp in zip(render_paths, target_paths):
        r = np.asarray(load_png(rp).permute(1, 2, 0).numpy() * 255, np.uint8)
        t = np.asarray(load_png(tp).permute(1, 2, 0).numpy() * 255, np.uint8)
        frames.append(np.concatenate([r, t], axis=1))  # render | target
    imageio.mimsave(path, frames, fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--render_sub", default="renders_native30")
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    runs = args.runs or f"{WT}/rebuttal/results/task_a_runs/{args.method}"
    tag = args.tag or args.render_sub
    ids = [args.only] if args.only else sorted(os.listdir(runs))
    rows = []
    for oid in ids:
        rdir = f"{runs}/{oid}/{args.render_sub}"
        bdir = f"{args.bundles}/{oid}"
        if not os.path.isdir(rdir):
            print(f"skip {oid} (no {args.render_sub})"); continue
        rp = [f"{rdir}/{f}" for f in sorted(os.listdir(rdir)) if f.endswith(".png")]
        tp = bundle_targets(bdir)
        n = min(len(rp), len(tp))
        m = score_views(rp[:n], tp[:n])
        m["object"] = oid
        rows.append(m)
        # montage + video
        viz = f"{runs}/{oid}/viz"
        os.makedirs(viz, exist_ok=True)
        try:
            orbit_video(rp[:n], tp[:n], f"{viz}/orbit_{tag}.mp4")
            inter = [x for pair in zip(rp[:8], tp[:8]) for x in pair]
            montage(inter, 4, f"{viz}/montage_{tag}.png")
        except Exception as e:
            print(f"viz fail {oid}: {e}")
        print(f"{oid}: PSNR {m['PSNR']:.2f} SSIM {m['SSIM']:.3f} LPIPS {m['LPIPS']:.3f}")
    if rows:
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ["PSNR", "SSIM", "LPIPS"]}
        out = f"{runs}/summary_{tag}.json"
        json.dump({"method": args.method, "tag": tag, "rows": rows, "mean": mean,
                   "n": len(rows)}, open(out, "w"), indent=1)
        print(f"\nMEAN (n={len(rows)}): {mean}\n-> {out}")


if __name__ == "__main__":
    main()
