"""FINV-SV geometry for ICP-Chamfer: run the multi-start particle inversion + pruning +
PTI, then extract a surface point cloud from the (PTI'd) EG3D density field. Same
extraction as eg3d_geom -> fair shape comparison. --realcars for task B."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
import finv_sv as FV
from eg3d_geom import extract_pointcloud
device = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_geom/finv")
    ap.add_argument("--n_particles", type=int, default=16)
    ap.add_argument("--realcars", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(device)
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    ids = [o for o in ids if not os.path.exists(f"{args.out}/{o}.npy")]
    inp = "rgb_white_128.png" if args.realcars else "input.png"
    for oid in ids:
        G = E.load_G()
        target = E.load_input(f"{args.bundles}/{oid}/{inp}")
        ws, c2w_hat, info = FV.finv_invert(G, lp, target, n_particles=args.n_particles)
        pc = extract_pointcloud(G, ws)
        if pc is None:
            print(f"FAIL {oid}"); del G; torch.cuda.empty_cache(); continue
        np.save(f"{args.out}/{oid}.npy", pc.astype(np.float32))
        print(f"done {oid} ({pc.shape[0]} pts) best_loss={info['best_loss']:.4f}")
        del G; torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
