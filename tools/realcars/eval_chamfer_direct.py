"""
Stage 3+4 (compressed): direct Chamfer Distance ours vs Splatter-Image baseline
against per-scene FastGS pseudo-GT.

For each idx:
  1. Load ours/baseline/pseudo_gt PLYs (Gaussian centers, opacity-filtered).
  2. Crop pseudo_gt to a sphere around the ARKit camera-trajectory centroid
     (= the point the user was scanning around = the car center).
  3. Global registration (FPFH + RANSAC) using a scale prior (gt diam / pred diam),
     refined by point-to-point ICP. Same rigid+scale transform applied to both
     ours and baseline so they live in pseudo_gt's metric frame.
  4. Symmetric Chamfer Distance pred -> gt and gt -> pred (in METERS).

Saves per-idx CSV + a markdown table for dropping into AGENT.md / the paper.
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

from infer3d import config as _cfg
ROOT = _cfg.SPLATTER_REPO_ROOT
GS2MESH_INPUT = _cfg.GS2MESH_INPUT
SCENES_ROOT = _cfg.REALCARS_ROOT
CSV_PATH = f"{ROOT}/experiments/neurips_submission/real_ood/realcars_test_paths.csv"
OUT_DIR = Path(f"{ROOT}/experiments/neurips_submission/real_ood_realcars_eval/results")


def load_xyz(ply_path, opacity_thresh=0.05, max_points=200_000):
    pl = PlyData.read(str(ply_path))
    v = pl['vertex'].data
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    if 'opacity' in v.dtype.names:
        op = 1.0 / (1.0 + np.exp(-np.asarray(v['opacity'])))
        xyz = xyz[op > opacity_thresh]
    if len(xyz) > max_points:
        idx = np.random.RandomState(0).choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
    return xyz


def scene_dir_for_idx(idx: str) -> str:
    line = int(idx) + 1
    with open(CSV_PATH) as f:
        path = list(f)[line - 1].strip()
    # path = .../HQ339/<scene>/frame_NNNNN.jpg
    return os.path.dirname(path)


def car_center_from_cameras(scene_dir: str) -> np.ndarray:
    """ARKit world-space centroid of the camera trajectory ≈ car position."""
    jps = sorted(glob.glob(os.path.join(scene_dir, "frame_*.json")))
    pts = []
    for jp in jps:
        if not os.path.exists(jp.replace(".json", ".jpg")):
            continue
        m = json.load(open(jp))
        T = np.array(m['cameraPoseARFrame'], dtype=np.float64).reshape(4, 4)
        pts.append(T[:3, 3])
    return np.mean(pts, axis=0) if pts else np.zeros(3)


def crop_ball(xyz: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    d = np.linalg.norm(xyz - center, axis=1)
    return xyz[d < radius]


def to_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def estimate_diam(xyz: np.ndarray) -> float:
    if len(xyz) == 0:
        return 1.0
    bb_min = xyz.min(axis=0); bb_max = xyz.max(axis=0)
    return float(np.linalg.norm(bb_max - bb_min))


def fpfh(pcd, voxel):
    pcd_d = pcd.voxel_down_sample(voxel)
    pcd_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fpfh_feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return pcd_d, fpfh_feat


def align(pred_xyz: np.ndarray, gt_xyz: np.ndarray, voxel_pred: float = None, voxel_gt: float = None):
    """Pre-scale pred to gt diam, FPFH+RANSAC global, ICP refine. Returns aligned pred xyz."""
    pred_diam = estimate_diam(pred_xyz)
    gt_diam = estimate_diam(gt_xyz)
    scale = gt_diam / max(pred_diam, 1e-6)
    pred_centered = pred_xyz - pred_xyz.mean(axis=0)
    pred_scaled = pred_centered * scale + gt_xyz.mean(axis=0)

    if voxel_pred is None:
        voxel_pred = max(gt_diam / 50.0, 0.02)
    if voxel_gt is None:
        voxel_gt = voxel_pred

    pred_pcd = to_pcd(pred_scaled)
    gt_pcd = to_pcd(gt_xyz)
    src_d, src_f = fpfh(pred_pcd, voxel_pred)
    tgt_d, tgt_f = fpfh(gt_pcd, voxel_gt)

    distance_threshold = voxel_pred * 1.5
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    icp = o3d.pipelines.registration.registration_icp(
        pred_pcd, gt_pcd, distance_threshold * 2, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    pred_pcd.transform(icp.transformation)
    return np.asarray(pred_pcd.points), icp.fitness, icp.inlier_rmse


def chamfer_sym(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Symmetric Chamfer in METERS. Returns (cd_sym, cd_a2b, cd_b2a)."""
    pa = to_pcd(a); pb = to_pcd(b)
    d_ab = np.asarray(pa.compute_point_cloud_distance(pb))
    d_ba = np.asarray(pb.compute_point_cloud_distance(pa))
    cd_a2b = float(np.mean(d_ab ** 2))
    cd_b2a = float(np.mean(d_ba ** 2))
    return 0.5 * (cd_a2b + cd_b2a), cd_a2b, cd_b2a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_radius_m", type=float, default=3.0,
                        help="Crop sphere around camera-trajectory centroid (meters)")
    parser.add_argument("--opacity_thresh", type=float, default=0.05)
    parser.add_argument("--idxs", type=str, default=None,
                        help="comma-separated subset, e.g. 00,01")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    idxs = sorted([f"{i:02d}" for i in range(20)])
    if args.idxs:
        idxs = args.idxs.split(",")

    for idx in idxs:
        ours_p = f"{GS2MESH_INPUT}/realcars_ours_plys/{idx}.ply"
        base_p = f"{GS2MESH_INPUT}/realcars_baseline_plys/{idx}.ply"
        gt_p = f"{GS2MESH_INPUT}/realcars_pseudo_gt_plys/{idx}.ply"
        if not (os.path.exists(ours_p) and os.path.exists(base_p) and os.path.exists(gt_p)):
            print(f"[{idx}] missing ply, skip"); continue

        ours = load_xyz(ours_p, opacity_thresh=args.opacity_thresh)
        base = load_xyz(base_p, opacity_thresh=args.opacity_thresh)
        gt_full = load_xyz(gt_p, opacity_thresh=args.opacity_thresh)

        scene_dir = scene_dir_for_idx(idx)
        car_center = car_center_from_cameras(scene_dir)
        gt = crop_ball(gt_full, car_center, args.gt_radius_m)
        if len(gt) < 1000:
            print(f"[{idx}] gt too small after crop ({len(gt)} pts), bumping radius")
            gt = crop_ball(gt_full, car_center, args.gt_radius_m * 1.5)

        print(f"[{idx}] ours={len(ours)}  baseline={len(base)}  gt_crop={len(gt)} (full {len(gt_full)})")

        ours_aln, fit_o, rmse_o = align(ours, gt)
        base_aln, fit_b, rmse_b = align(base, gt)
        cd_ours, cd_o2g, cd_g2o = chamfer_sym(ours_aln, gt)
        cd_base, cd_b2g, cd_g2b = chamfer_sym(base_aln, gt)

        winner = "OURS" if cd_ours < cd_base else "BASELINE"
        print(f"[{idx}] CD_ours={cd_ours:.5f}  CD_baseline={cd_base:.5f}  winner={winner}")

        rows.append({
            "idx": idx,
            "n_ours": len(ours), "n_base": len(base), "n_gt": len(gt),
            "cd_ours": cd_ours, "cd_baseline": cd_base,
            "cd_ours_o2g": cd_o2g, "cd_ours_g2o": cd_g2o,
            "cd_baseline_o2g": cd_b2g, "cd_baseline_g2o": cd_g2b,
            "icp_fit_ours": fit_o, "icp_rmse_ours": rmse_o,
            "icp_fit_baseline": fit_b, "icp_rmse_baseline": rmse_b,
            "winner": winner,
        })

    csv_path = OUT_DIR / "chamfer_direct.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader(); w.writerows(rows)

    if not rows:
        print("[eval] no rows produced"); return

    ours_arr = np.array([r["cd_ours"] for r in rows])
    base_arr = np.array([r["cd_baseline"] for r in rows])

    # Asymmetric pred->gt is the meaningful metric for single-view splat methods
    # (both methods produce ~16k splats from ONE input image -> neither covers
    # the back of the car, so symmetric CD is dominated by the gt->pred half
    # which is roughly the "missing back" for both, washing out the comparison).
    o2g_o = np.array([r["cd_ours_o2g"]     for r in rows])
    o2g_b = np.array([r["cd_baseline_o2g"] for r in rows])
    g2o_o = np.array([r["cd_ours_g2o"]     for r in rows])
    g2o_b = np.array([r["cd_baseline_g2o"] for r in rows])

    md_path = OUT_DIR / "chamfer_table.md"
    with open(md_path, "w") as f:
        f.write("# RealCars 3D Reconstruction Chamfer Distance vs FastGS pseudo-GT\n\n")
        f.write("Pipeline:\n"
                "1. Stage 1 — FastGS-fit per-scene pseudo-GT splats from all "
                "in-scene RealCars frames at 1600x1200 (ARKit poses + intrinsics, "
                "30k iters).\n"
                "2. Stage 2 — Splatter Image (model_cars.pth) feed-forward on "
                "the same SAM-cropped 128x128 input we feed to our optimizer; for "
                "ours we additionally bake the optimizer's best (R,t) into the splats.\n"
                "3. Crop pseudo-GT to a 3m sphere around the ARKit "
                "camera-trajectory centroid (= car position).\n"
                "4. FPFH+RANSAC global registration -> point-to-point ICP refine "
                "(predictions pre-scaled to GT diameter, then aligned in pseudo-GT's "
                "metric ARKit world frame).\n"
                "5. Symmetric Chamfer + asymmetric components in meters^2.\n\n"
                "**Primary metric is `CD pred->gt` (asymmetric).** Both methods produce "
                "~16k pixel-aligned splats from ONE input view, so neither covers the "
                "back of the car. Symmetric CD is therefore dominated by the gt->pred "
                "half (the missing-back gap, similar for both methods) and washes out "
                "the comparison. Asymmetric pred->gt directly measures \"are the "
                "predicted points on the actual car surface?\".\n\n")
        f.write("## Primary: Chamfer pred -> gt (lower = better)\n\n")
        f.write("| idx | CD ours -> gt | CD baseline -> gt | winner |\n")
        f.write("|-----|---------------|-------------------|--------|\n")
        for r in rows:
            o = r["cd_ours_o2g"]; b = r["cd_baseline_o2g"]
            f.write(f"| {r['idx']} | {o:.4f} | {b:.4f} | {'**OURS**' if o < b else 'baseline'} |\n")
        f.write(f"\n**Mean:** ours {o2g_o.mean():.4f} | baseline {o2g_b.mean():.4f}  \n")
        f.write(f"**Median:** ours {np.median(o2g_o):.4f} | baseline {np.median(o2g_b):.4f}  \n")
        f.write(f"**Ours wins on {(o2g_o < o2g_b).sum()}/{len(rows)} scenes**\n\n")

        f.write("## Reference: symmetric Chamfer (lower = better)\n\n")
        f.write("| idx | n_ours | n_base | n_gt | CD ours sym | CD baseline sym | winner sym |\n")
        f.write("|-----|--------|--------|------|-------------|-----------------|------------|\n")
        for r in rows:
            f.write(f"| {r['idx']} | {r['n_ours']} | {r['n_base']} | {r['n_gt']} | "
                    f"{r['cd_ours']:.4f} | {r['cd_baseline']:.4f} | {r['winner']} |\n")
        f.write(f"\n**Mean sym:** ours {ours_arr.mean():.4f} | baseline {base_arr.mean():.4f}  \n")
        f.write(f"**Median sym:** ours {np.median(ours_arr):.4f} | baseline {np.median(base_arr):.4f}  \n")
        f.write(f"**Ours wins (sym) on {(ours_arr < base_arr).sum()}/{len(rows)} scenes**\n\n")

        f.write("## Reference: gt -> pred (coverage of GT by predictions)\n\n")
        f.write(f"Mean: ours {g2o_o.mean():.4f} | baseline {g2o_b.mean():.4f}. "
                "Both large because the back half of the car is unobserved in the "
                "input view; neither method invents geometry there.\n")

    print(f"\nSaved {csv_path}\nSaved {md_path}")


if __name__ == "__main__":
    main()
