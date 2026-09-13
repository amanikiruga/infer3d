"""Decisive test: enumerate all 24 proper-rotation local-axis conventions for the
oracle camera mapping; score mean PSNR over all views. If the best reaches NFI's
control (~19), it's a findable convention; if stuck ~14, it's reconstruction quality
or an ICP global-rotation branch ambiguity (not pose-convention)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, sys, itertools
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import nfi_task_a as N
import nfi_geom as NG
from icp_chamfer import align_transform
from scorer import score_views, bundle_targets
dev = "cuda"
GTDIR = f"{WT}/rebuttal/results/task_a_geom/gt"
oid = sys.argv[1] if len(sys.argv) > 1 else "1ae184691a39e3d3e0e8bce75d28b114"
bdir = f"{WT}/rebuttal/results/task_a_inputs/{oid}"


def proper_rotations():
    """24 proper rotation matrices (octahedral group) as signed axis permutations."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1) < 1e-6:
                mats.append(M)
    return mats


G, enc, lp = N.load_models()
cams = np.load(f"{bdir}/cams_relative.npz"); fov = float(cams["fov_deg"])
focal = (N.RES / 2) / np.tan(np.deg2rad(fov) / 2) / N.RES
cabs = np.load(f"{bdir}/cams_absolute.npz"); targets_v2w = cabs["targets_v2w"]
torch.manual_seed(0); np.random.seed(0)
ws, cam, foc, _ = N.invert(G, enc, lp, N.load_input(f"{bdir}/input.png"), 300, focal)
pc = NG.extract_pc(G, ws)
gt = np.load(f"{GTDIR}/{oid}.npy")
s, R, t, fit, resid = align_transform(pc.astype(np.float64), gt.astype(np.float64))
print(f"ICP fit={fit:.3f} s={s:.3f}")
foc_gt = torch.tensor([focal], device=dev).float()
tps = bundle_targets(bdir)
mats = proper_rotations()
Rt = R.T
# subsample views for speed during search
view_idx = [0, 3, 6, 9, 12, 15, 18, 21]
tps_sub = [tps[i] for i in view_idx]
best = None
for k, P in enumerate(mats):
    P4 = np.eye(4); P4[:3, :3] = P
    rpaths = []
    rd = f"{WT}/rebuttal/results/_perm/{k}"; os.makedirs(rd, exist_ok=True)
    for vi in view_idx:
        C = targets_v2w[vi].T.astype(np.float64); Rc, c = C[:3, :3], C[:3, 3]
        M = np.eye(4, dtype=np.float32); M[:3, :3] = Rt @ Rc; M[:3, 3] = (1.0 / s) * (Rt @ (c - t))
        M = (M @ P4).astype(np.float32)
        with torch.no_grad():
            rgb, _, _, _ = N.render(G, N.RES, N.RES, torch.from_numpy(M)[None].to(dev), foc_gt, ws, randomize=False)
        p = f"{rd}/{vi:02d}.png"; N.save_rgb(torch.clamp(rgb[0], -1, 1), p); rpaths.append(p)
    psnr = score_views(rpaths, tps_sub)["PSNR"]
    if best is None or psnr > best[0]:
        best = (psnr, k, P)
        print(f"  new best perm#{k} PSNR={psnr:.2f}\n{P.astype(int)}")
print("BEST", best[0], "perm#", best[1])
print(best[2].astype(int))
