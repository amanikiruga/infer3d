"""EG3D-PTI geometry for Task A SO(3) cars — GAUGE-FREE (ICP-aligned Chamfer).

For each object: invert the pretrained EG3D shapenet-cars generator from the single
SO(3) input (pose optimized = the method's own no-GT-pose mode; ICP later removes
any residual gauge), PTI fine-tune, then extract a surface point cloud from the
density field via marching cubes. Chamfer vs GT ShapeNet mesh is scored separately
(icp_chamfer.py), so camera convention is irrelevant here.

  python eg3d_geom.py [--only OBJ] [--bundles ...] [--out ...]
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys, json
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
import lpips as lpips_lib
from skimage import measure

dev = "cuda"


def extract_pointcloud(G, ws, N=96, n_surface=50000, level=10.0):
    bw = G.rendering_kwargs["box_warp"]
    g = torch.linspace(-bw / 2, bw / 2, N, device=dev)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(1, -1, 3)
    sig = []
    with torch.no_grad():
        for i in range(0, pts.shape[1], 200000):
            chunk = pts[:, i:i+200000]
            s = G.sample_mixed(chunk, torch.zeros_like(chunk), ws, noise_mode="const")["sigma"]
            sig.append(s.squeeze(0).squeeze(-1))
    sig = torch.cat(sig).reshape(N, N, N).cpu().numpy()
    if not (sig.min() < level < sig.max()):
        level = float(np.median(sig[sig > sig.mean()]))  # fallback
    try:
        verts, faces, _, _ = measure.marching_cubes(sig, level=level)
    except Exception:
        return None
    # verts in voxel idx -> world coords
    verts = verts / (N - 1) * bw - bw / 2
    # sample surface points from the mesh faces (area-weighted)
    import trimesh
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if m.area == 0:
        return None
    p, _ = trimesh.sample.sample_surface(m, n_surface)
    return np.asarray(p, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_geom/eg3d")
    ap.add_argument("--only", default=None)
    ap.add_argument("--main_steps", type=int, default=300)
    ap.add_argument("--pti_steps", type=int, default=150)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--realcars", action="store_true")
    args = ap.parse_args()
    lp = lpips_lib.LPIPS(net="vgg").to(dev)
    ids = [args.only] if args.only else sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]  # shard
    ids = [o for o in ids if not os.path.exists(f"{args.out}/{o}.npy")]      # resume
    os.makedirs(args.out, exist_ok=True)
    for oid in ids:
        G = E.load_G()
        inp = "rgb_white_128.png" if args.realcars else "input.png"
        target = E.load_input(f"{args.bundles}/{oid}/{inp}")
        ws, _, _ = E.invert(G, lp, target, main_steps=args.main_steps, pti_steps=args.pti_steps)
        pc = extract_pointcloud(G, ws)
        if pc is None:
            print(f"FAIL geom {oid}"); del G; torch.cuda.empty_cache(); continue
        np.save(f"{args.out}/{oid}.npy", pc)
        print(f"done {oid}  ({pc.shape[0]} pts)")
        del G; torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
