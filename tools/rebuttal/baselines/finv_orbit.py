"""Free azimuth orbit (36 frames) of the FINV-SV reconstruction for the gallery.
Runs the multi-start particle inversion + PTI, then orbits the winning EG3D reconstruction
(same orbit convention as eg3d_orbit)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
import finv_sv as FV
from utils.camera_utils import LookAtPoseSampler as LP
dev = "cuda"; NF = 36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/orbits/finv_so3")
    ap.add_argument("--realcars", action="store_true")
    ap.add_argument("--n_particles", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(dev)
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    ids = [o for o in ids if not os.path.exists(f"{args.out}/{o}.npy")]
    inp = "rgb_white_128.png" if args.realcars else "input.png"
    for oid in ids:
        G = E.load_G()
        target = E.load_input(f"{args.bundles}/{oid}/{inp}")
        ws, _, _ = FV.finv_invert(G, lp, target, n_particles=args.n_particles)
        r = G.rendering_kwargs.get("avg_camera_radius", 1.7)
        piv = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=dev)
        frames = []
        for az in np.linspace(0, 2 * np.pi, NF, endpoint=False):
            c2w = LP.sample(float(az), np.pi / 2 - 0.35, piv, radius=r, device=dev)
            c = torch.cat([c2w.reshape(1, 16), E.INTR.to(dev).reshape(1, 9)], 1)
            with torch.no_grad():
                img = E.synth(G, ws, c)[0]
            frames.append(np.clip((img.permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8))
        np.save(f"{args.out}/{oid}.npy", np.stack(frames)); print(f"done {oid}")
        del G; torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
