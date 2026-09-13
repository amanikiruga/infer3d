"""Sample GT surface point clouds from ShapeNet model_normalized.obj for bench25 cars."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os
import numpy as np, trimesh

WT = f"{_EXT}/splatter-image-rebuttal"
SN = f"{_EXT}/datasets/ShapeNetCore.v2/02958343"
OUT = f"{WT}/rebuttal/results/task_a_geom/gt"
os.makedirs(OUT, exist_ok=True)

ids = [e[0] for e in json.load(open(f"{WT}/rebuttal/results/task_a_bench25.json"))]
for o in ids:
    p = f"{SN}/{o}/models/model_normalized.obj"
    m = trimesh.load(p, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(m, 50000)
    np.save(f"{OUT}/{o}.npy", np.asarray(pts, np.float32))
    print(f"{o}: {len(pts)} pts, diam={np.linalg.norm(pts.max(0)-pts.min(0)):.3f}")
print("GT DONE")
