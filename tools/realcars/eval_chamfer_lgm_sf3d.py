"""
Same Chamfer pipeline as eval_chamfer_direct.py, but for LGM (Gaussian PLY) and
SF3D (glb mesh -> sampled surface points). Produces a 4-method combined table
(ours / Splatter Image / LGM / SF3D) using the same ours+baseline numbers from
chamfer_direct.csv.

Pipeline per idx:
  1. Load LGM gaussian centers (opacity-filtered) and SF3D mesh surface samples.
  2. Crop pseudo_gt to a sphere around the ARKit camera-trajectory centroid.
  3. Pre-scale pred to GT diameter, FPFH+RANSAC global -> ICP refine.
  4. Asymmetric Chamfer pred->gt in meters^2.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData

from eval_chamfer_direct import (
    SCENES_ROOT, CSV_PATH, GS2MESH_INPUT, OUT_DIR,
    load_xyz, scene_dir_for_idx, car_center_from_cameras,
    crop_ball, align, chamfer_sym,
)

from infer3d import config as _cfg
ROOT = _cfg.SPLATTER_REPO_ROOT
LGM_DIR = f"{ROOT}/experiments/neurips_submission/real_ood_realcars_LGM"
SF3D_DIR = f"{ROOT}/experiments/neurips_submission/real_ood_realcars_SF3D"


def load_lgm_xyz(idx, opacity_thresh=0.05, max_points=200_000):
    p = f"{LGM_DIR}/{idx}/lgm.ply"
    if not os.path.exists(p):
        return None
    return load_xyz(p, opacity_thresh=opacity_thresh, max_points=max_points)


def load_sf3d_xyz(idx, n_samples=50000):
    p = f"{SF3D_DIR}/{idx}/mesh.glb"
    if not os.path.exists(p):
        return None
    m = trimesh.load(p, force='mesh')
    if len(m.faces) == 0 or len(m.vertices) == 0:
        return None
    pts, _ = trimesh.sample.sample_surface(m, n_samples)
    return np.asarray(pts, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_radius_m", type=float, default=3.0)
    parser.add_argument("--opacity_thresh", type=float, default=0.05)
    parser.add_argument("--idxs", type=str, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    idxs = sorted([f"{i:02d}" for i in range(20)])
    if args.idxs:
        idxs = args.idxs.split(",")

    for idx in idxs:
        gt_p = f"{GS2MESH_INPUT}/realcars_pseudo_gt_plys/{idx}.ply"
        if not os.path.exists(gt_p):
            print(f"[{idx}] missing gt ply, skip"); continue

        gt_full = load_xyz(gt_p, opacity_thresh=args.opacity_thresh)
        scene_dir = scene_dir_for_idx(idx)
        car_center = car_center_from_cameras(scene_dir)
        gt = crop_ball(gt_full, car_center, args.gt_radius_m)
        if len(gt) < 1000:
            gt = crop_ball(gt_full, car_center, args.gt_radius_m * 1.5)

        out = {"idx": idx, "n_gt": len(gt)}

        for name, loader in [("lgm", lambda: load_lgm_xyz(idx, args.opacity_thresh)),
                             ("sf3d", lambda: load_sf3d_xyz(idx))]:
            pred = loader()
            if pred is None or len(pred) < 100:
                print(f"[{idx}] {name}: missing/empty")
                out[f"n_{name}"] = 0
                out[f"cd_{name}_p2g"] = float('nan')
                out[f"cd_{name}_sym"] = float('nan')
                out[f"cd_{name}_g2p"] = float('nan')
                continue
            try:
                pred_aln, fit, rmse = align(pred, gt)
                cd_sym, cd_p2g, cd_g2p = chamfer_sym(pred_aln, gt)
                print(f"[{idx}] {name}: n={len(pred)} CD_p2g={cd_p2g:.5f} sym={cd_sym:.5f} fit={fit:.3f}")
                out[f"n_{name}"] = len(pred)
                out[f"cd_{name}_p2g"] = cd_p2g
                out[f"cd_{name}_sym"] = cd_sym
                out[f"cd_{name}_g2p"] = cd_g2p
            except Exception as e:
                print(f"[{idx}] {name}: align/cd FAILED: {e}")
                out[f"n_{name}"] = len(pred)
                out[f"cd_{name}_p2g"] = float('nan')
                out[f"cd_{name}_sym"] = float('nan')
                out[f"cd_{name}_g2p"] = float('nan')

        rows.append(out)

    csv_path = OUT_DIR / "chamfer_lgm_sf3d.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
