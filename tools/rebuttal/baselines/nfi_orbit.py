"""Free azimuth orbit (36 frames) of the nerf-from-image reconstruction for the gallery."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, math, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import nfi_task_a as N
dev = "cuda"; NF = 36


def orbit_c2w(az, el, r):
    # OpenGL cam2world looking at origin (NFI convention: x right, y up, -z fwd)
    x = r * math.cos(el) * math.cos(az); y = r * math.sin(el); z = r * math.cos(el) * math.sin(az)
    eye = np.array([x, y, z]); fwd = -eye / np.linalg.norm(eye)   # -z points to origin
    up = np.array([0, 1., 0])
    if abs(np.dot(fwd, up)) > .99: up = np.array([0, 0, 1.])
    right = np.cross(up, -fwd); right /= np.linalg.norm(right); up2 = np.cross(-fwd, right)
    c2w = np.eye(4, dtype=np.float32); c2w[:3, 0] = right; c2w[:3, 1] = up2; c2w[:3, 2] = -fwd; c2w[:3, 3] = eye
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/orbits/nfi_so3")
    ap.add_argument("--realcars", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    G, enc, lp = N.load_models()
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    ids = [o for o in ids if not os.path.exists(f"{args.out}/{o}.npy")]
    inp = "rgb_white_128.png" if args.realcars else "input.png"
    for oid in ids:
        bdir = f"{args.bundles}/{oid}"
        if args.realcars:
            fg = 131.25 / 128.
        else:
            fg = (N.RES / 2) / np.tan(np.deg2rad(float(np.load(f"{bdir}/cams_relative.npz")["fov_deg"])) / 2) / N.RES
        torch.manual_seed(0); np.random.seed(0)
        ws, _, foc, _ = N.invert(G, enc, lp, N.load_input(f"{bdir}/{inp}"), args.steps, fg)
        frames = []
        for az in np.linspace(0, 2 * np.pi, NF, endpoint=False):
            c2w = torch.from_numpy(orbit_c2w(float(az), 0.35, 1.3))[None].to(dev)
            with torch.no_grad():
                rgb, _, _, _ = N.render(G, N.RES, N.RES, c2w, foc, ws, randomize=False)
            frames.append(np.clip((rgb[0].cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8))
        np.save(f"{args.out}/{oid}.npy", np.stack(frames)); print(f"done {oid}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
