"""Free azimuth orbit (36 frames) of the pi-GAN reconstruction, for the video gallery."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, math, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import pigan_geom as P
dev = "cuda"; NF = 36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/orbits/pigan_so3")
    ap.add_argument("--realcars", action="store_true")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    G = P.load_gen()
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    ids = [o for o in ids if not os.path.exists(f"{args.out}/{o}.npy")]
    inp = "rgb_white_128.png" if args.realcars else "input.png"
    for oid in ids:
        wf, wp = P.invert(G, P.load_img(f"{args.bundles}/{oid}/{inp}"), args.iters)
        frames = []
        for az in np.linspace(0, 2 * np.pi, NF, endpoint=False):
            o = dict(P.OPT); o["h_mean"] = float(az); o["v_mean"] = math.pi / 2 - 0.35
            o["hierarchical_sample"] = True; o["num_steps"] = 48; o["lock_view_dependence"] = True
            with torch.no_grad():
                px, _ = G.staged_forward_with_frequencies(wf, wp, max_batch_size=400000, **o)
            img = px[0].permute(1, 2, 0).cpu().numpy()
            frames.append(np.clip((img / 2 + 0.5) * 255, 0, 255).astype(np.uint8))
        np.save(f"{args.out}/{oid}.npy", np.stack(frames))
        print(f"done {oid}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
