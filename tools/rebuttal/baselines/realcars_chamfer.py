"""RealCars Chamfer for a baseline's predicted point clouds vs LiDAR pseudo-GT.

Reuses the paper's realcars eval (crop GT to ARKit car-center ball -> pre-scale ->
FPFH+RANSAC -> ICP -> Chamfer in meters^2), identical alignment to ours/Splatter/
LGM/SF3D. Pred = {idx}.npy from a baseline geometry extractor.

  python realcars_chamfer.py --pred_dir <dir of {idx}.npy> --tag <name>
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, csv, json, os, sys
import numpy as np

MAIN = f"{_EXT}/splatter-image"
EVALDIR = f"{MAIN}/experiments/neurips_submission/real_ood_realcars_eval"
sys.path.insert(0, EVALDIR)
import eval_chamfer_direct as EC  # crop_ball, car_center_from_cameras, scene_dir_for_idx, load_xyz, align, chamfer_sym

WT = f"{_EXT}/splatter-image-rebuttal"
GS2MESH_INPUT = f"{_EXT}/gs2mesh/input"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gt_radius_m", type=float, default=3.0)
    args = ap.parse_args()
    rows = []
    for idx in [f"{i:02d}" for i in range(20)]:
        pp = f"{args.pred_dir}/{idx}.npy"
        gt_p = f"{GS2MESH_INPUT}/realcars_pseudo_gt_plys/{idx}.ply"
        if not (os.path.exists(pp) and os.path.exists(gt_p)):
            continue
        pred = np.load(pp).astype(np.float64)
        gt_full = EC.load_xyz(gt_p, opacity_thresh=0.05)
        try:
            center = EC.car_center_from_cameras(EC.scene_dir_for_idx(idx))
            gt = EC.crop_ball(gt_full, center, args.gt_radius_m)
            if len(gt) < 1000:
                gt = EC.crop_ball(gt_full, center, args.gt_radius_m * 1.5)
            aln, fit, rmse = EC.align(pred, gt)
            cds, ca, cb = EC.chamfer_sym(aln, gt)
        except Exception as e:
            print(f"[{idx}] FAIL {e}"); continue
        rows.append({"idx": idx, "cd_sym": cds, "cd_pred2gt": ca, "cd_gt2pred": cb,
                     "icp_fit": fit, "n_pred": len(pred), "n_gt": len(gt)})
        print(f"[{idx}] cd_sym={cds:.4f} pred2gt={ca:.4f} fit={fit:.2f}")
    if not rows:
        print("no rows"); return
    cds = np.array([r["cd_sym"] for r in rows]); p2g = np.array([r["cd_pred2gt"] for r in rows])
    out = f"{WT}/rebuttal/results/task_b_geom/chamfer_{args.tag}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = {"tag": args.tag, "n": len(rows), "cd_sym_mean": float(cds.mean()),
            "cd_sym_median": float(np.median(cds)), "cd_pred2gt_mean": float(p2g.mean()),
            "cd_pred2gt_median": float(np.median(p2g))}
    json.dump(summ, open(out.replace(".csv", "_summary.json"), "w"), indent=1)
    print(f"\n[{args.tag}] n={len(rows)} cd_sym mean={cds.mean():.4f} median={np.median(cds):.4f} "
          f"| pred2gt mean={p2g.mean():.4f}")


if __name__ == "__main__":
    main()
