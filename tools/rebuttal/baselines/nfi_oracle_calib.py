"""Calibrate the oracle camera convention: try variants of the GT-cam -> NFI-frame
mapping, render all 24 views for one object, report mean PSNR. Pick the best variant."""
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

G, enc, lp = N.load_models()
cams = np.load(f"{bdir}/cams_relative.npz"); fov = float(cams["fov_deg"])
focal = (N.RES / 2) / np.tan(np.deg2rad(fov) / 2) / N.RES
cabs = np.load(f"{bdir}/cams_absolute.npz"); targets_v2w = cabs["targets_v2w"]
torch.manual_seed(0); np.random.seed(0)
ws, cam, foc, _ = N.invert(G, enc, lp, N.load_input(f"{bdir}/input.png"), 300, focal)
pc = NG.extract_pc(G, ws)
gt = np.load(f"{GTDIR}/{oid}.npy")
s, R, t, fit, resid = align_transform(pc.astype(np.float64), gt.astype(np.float64))
print(f"ICP fit={fit:.3f} s={s:.3f} resid={resid:.2e}")
foc_gt = torch.tensor([focal], device=dev).float()
FLIP = N.FLIP
tps = bundle_targets(bdir)


def render_variant(useRT, flipmode):
    Ruse = R.T if useRT else R
    rpaths = []
    rd = f"{WT}/rebuttal/results/_calib/{oid}/{useRT}_{flipmode}"; os.makedirs(rd, exist_ok=True)
    for i in range(targets_v2w.shape[0]):
        C = targets_v2w[i].T.astype(np.float64)             # GT target cam2world CV
        Rc, c = C[:3, :3], C[:3, 3]
        Rc_m = Ruse @ Rc
        c_m = (1.0 / s) * (Ruse @ (c - t))
        M = np.eye(4, dtype=np.float32); M[:3, :3] = Rc_m; M[:3, 3] = c_m
        if flipmode == "right": M = M @ FLIP
        elif flipmode == "left": M = FLIP @ M
        elif flipmode == "both": M = FLIP @ M @ FLIP
        with torch.no_grad():
            rgb, _, _, _ = N.render(G, N.RES, N.RES, torch.from_numpy(M)[None].to(dev), foc_gt, ws, randomize=False)
        p = f"{rd}/{i:02d}.png"; N.save_rgb(torch.clamp(rgb[0], -1, 1), p); rpaths.append(p)
    return score_views(rpaths, tps)["PSNR"]


best = None
for useRT, flipmode in itertools.product([True, False], ["none", "right", "left", "both"]):
    psnr = render_variant(useRT, flipmode)
    tag = f"{'R^T' if useRT else 'R'}/{flipmode}"
    print(f"  {tag:12s} meanPSNR={psnr:.2f}")
    if best is None or psnr > best[0]: best = (psnr, tag)
print("BEST", best)
