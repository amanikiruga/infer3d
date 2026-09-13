"""Consolidate per-object SO(3) NVS PSNR for every method into one JSON, for the report
(scores shown under each per-object video). PSNR only (CPU; no GPU/LPIPS needed)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, json, os
import numpy as np
from PIL import Image
WT = f"{_EXT}/splatter-image-rebuttal"
R = f"{WT}/rebuttal/results"; TGT = f"{R}/task_a_inputs"


def psnr_dir(rdir, oid):
    tdir = f"{TGT}/{oid}/targets"
    if not os.path.isdir(rdir) or not os.path.isdir(tdir):
        return None
    tfs = sorted(os.listdir(tdir)); ps = []
    rfs = sorted(glob.glob(f"{rdir}/*.png"))
    for i, rf in enumerate(rfs):
        if i >= len(tfs):
            break
        a = np.asarray(Image.open(rf).convert("RGB").resize((128, 128))).astype(np.float32) / 255
        b = np.asarray(Image.open(f"{tdir}/{tfs[i]}").convert("RGB").resize((128, 128))).astype(np.float32) / 255
        mse = np.mean((a - b) ** 2)
        ps.append(-10 * np.log10(mse + 1e-10))
    return float(np.mean(ps)) if ps else None


scores = {}
# ours from summary.json
oursj = json.load(open(f"{R}/task_a_runs/ours_nvs/summary.json"))
for row in oursj["rows"]:
    scores.setdefault(row["object"], {})["ours"] = round(row["PSNR"], 2)
# oracle metrics from json
for tag, sub in [("nfi_oracle", "task_a_oracle/nfi_so3"), ("eg3d_oracle", "task_a_oracle/eg3d")]:
    for mp in glob.glob(f"{R}/{sub}/*/metrics.json"):
        oid = os.path.basename(os.path.dirname(mp))
        scores.setdefault(oid, {})[tag] = round(json.load(open(mp))["PSNR"], 2)
# own-pose render dirs scored on CPU
for tag, patt in [("nfi_own", "task_a_runs/nfi/{oid}/renders_ext300"),
                  ("eg3d_own", "task_a_runs/eg3d/{oid}/renders"),
                  ("finv_own", "task_a_runs/finv/{oid}/renders")]:
    for oid in os.listdir(TGT):
        p = psnr_dir(f"{R}/{patt.format(oid=oid)}", oid)
        if p is not None:
            scores.setdefault(oid, {})[tag] = round(p, 2)
json.dump(scores, open(f"{R}/per_object_scores.json", "w"), indent=1)
# aggregates
keys = ["ours", "nfi_own", "nfi_oracle", "eg3d_own", "eg3d_oracle", "finv_own"]
print("PER-METHOD AGGREGATE (SO(3) NVS PSNR):")
for k in keys:
    vals = [v[k] for v in scores.values() if k in v]
    if vals:
        print(f"  {k:12s} n={len(vals):2d}  mean={np.mean(vals):.2f}")
print(f"wrote per_object_scores.json ({len(scores)} objects)")
