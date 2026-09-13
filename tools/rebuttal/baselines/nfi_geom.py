"""nerf-from-image geometry for Task A (SO(3) cars) — gauge-free ICP-Chamfer.

Invert (encoder+PnP+latent opt, no GT pose) then extract a surface point cloud
from the SDF field (sampler 'sdf_distance', level 0). Camera frame irrelevant
(ICP aligns). --bundles selects so3 or control inputs.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch, trimesh
from skimage import measure
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import nfi_task_a as N
dev = "cuda"


def extract_pc(G, ws, N_grid=96, n_surface=50000):
    r = 0.55  # scene_range
    g = torch.linspace(-r, r, N_grid, device=dev)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(1, -1, 1, 3)
    sampler = G(None, ws, ["sampler"])["sampler"]
    sdf = []
    with torch.no_grad():
        for i in range(0, pts.shape[1], 200000):
            out = sampler(pts[:, i:i+200000], ["sdf_distance"])
            sdf.append(out["sdf_distance"].reshape(-1))
    sdf = torch.cat(sdf).reshape(N_grid, N_grid, N_grid).cpu().numpy()
    if not (sdf.min() < 0 < sdf.max()):
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)
    except Exception:
        return None
    verts = verts / (N_grid - 1) * 2 * r - r
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if m.area == 0:
        return None
    p, _ = trimesh.sample.sample_surface(m, n_surface)
    return np.asarray(p, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_geom/nfi")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--realcars", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    G, enc, lp = N.load_models()
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    ids = [o for o in ids if not os.path.exists(f"{args.out}/{o}.npy")]
    for oid in ids:
        bdir = f"{args.bundles}/{oid}"
        if args.realcars:
            inp = f"{bdir}/rgb_white_128.png"; fg = 131.25 / 128.  # SRN-cars focal (ICP normalizes scale)
        else:
            cams = np.load(f"{bdir}/cams_relative.npz")
            fov = float(cams["fov_deg"]); fg = (N.RES / 2) / np.tan(np.deg2rad(fov) / 2) / N.RES
            inp = f"{bdir}/input.png"
        torch.manual_seed(0); np.random.seed(0)
        ws, _, _, _ = N.invert(G, enc, lp, N.load_input(inp), args.steps, fg)
        pc = extract_pc(G, ws)
        if pc is None:
            print(f"FAIL {oid}"); continue
        np.save(f"{args.out}/{oid}.npy", pc)
        print(f"done {oid} ({pc.shape[0]} pts)")
    print("ALL DONE")


if __name__ == "__main__":
    main()
