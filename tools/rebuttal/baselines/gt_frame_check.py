"""GT-only sanity: project the GT point cloud through the exported target cameras and
overlay on the GT target images. If aligned, GT-geom frame == camera frame (targets_v2w).
Uses the splatter camera builder from view_to_world (absolute)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, sys
import numpy as np
from PIL import Image
WT = f"{_EXT}/splatter-image-rebuttal"
oid = sys.argv[1] if len(sys.argv) > 1 else "1ae184691a39e3d3e0e8bce75d28b114"
R = f"{WT}/rebuttal/results"
gt = np.load(f"{R}/task_a_geom/gt/{oid}.npy")          # (N,3)
cabs = np.load(f"{R}/task_a_inputs/{oid}/cams_absolute.npz")
cams = np.load(f"{R}/task_a_inputs/{oid}/cams_relative.npz")
fov = float(cams["fov_deg"]); RESO = int(cams["resolution"])
tdir = f"{R}/task_a_inputs/{oid}/targets"; tfs = sorted(os.listdir(tdir))
f = (RESO / 2) / np.tan(np.deg2rad(fov) / 2)
rows = []
for i in [0, 4, 8, 12, 16, 20]:
    c2w = cabs["targets_v2w"][i].T.astype(np.float64)   # CV cam2world
    w2c = np.linalg.inv(c2w)
    X = np.concatenate([gt, np.ones((len(gt), 1))], 1)   # (N,4)
    Xc = (w2c @ X.T).T[:, :3]                             # camera coords (CV: +z forward)
    z = Xc[:, 2]
    valid = z > 1e-4
    u = f * Xc[:, 0] / z + RESO / 2
    v = f * Xc[:, 1] / z + RESO / 2
    img = np.asarray(Image.open(f"{tdir}/{tfs[i]}").convert("RGB").resize((RESO, RESO))).copy()
    uu = np.round(u[valid]).astype(int); vv = np.round(v[valid]).astype(int)
    m = (uu >= 0) & (uu < RESO) & (vv >= 0) & (vv < RESO)
    img[vv[m], uu[m]] = [255, 0, 0]                       # project GT pts as red
    rows.append(img)
Image.fromarray(np.concatenate(rows, 1)).save(f"{R}/gt_frame_check.png")
print("saved gt_frame_check.png (red = GT points projected via targets_v2w over GT target image)")
