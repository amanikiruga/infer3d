"""Scale-SENSITIVE Chamfer LOO (companion to chamfer_loo.py).

L_depth constrains ABSOLUTE metric scale/depth — which the similarity (scale-removing)
ICP in chamfer_loo.py explicitly factors out. Here alignment is RIGID only (rotation +
translation via centroid shift -> FPFH+RANSAC -> point-to-point ICP, NO pre-scaling, NO
scale estimation), so a wrong absolute scale shows up in the CD. Reuses the cached
per-arm point clouds; pseudo-GT and predictions share the CO3D world scale (zgt/camera
geometry), so scale is commensurable.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, json, os, sys
import numpy as np

WT = f"{_EXT}/splatter-image-rebuttal"
ORIG = f"{_EXT}/splatter-image"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
from icp_chamfer import to_pcd, fpfh, chamfer_sym, estimate_diam  # noqa: E402
import open3d as o3d  # noqa: E402
from plyfile import PlyData  # noqa: E402

ARMS = ["b0_control", "no_depth", "no_latent_prior"]
OUT = f"{WT}/rebuttal/results/ablation_runs_co3d/chamfer_loo"
GT_LIST = f"{ORIG}/eval_output/checkpoints-icml-diffae-co3d-hydrants-baseline-se3-new-stylegan-10-depth-ood-final/lists/pseudo_gt.txt"


def load_ply_xyz(path, n=50000):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if len(xyz) > n:
        xyz = xyz[np.random.RandomState(0).choice(len(xyz), n, replace=False)]
    return xyz


def align_rigid(pred_xyz, gt_xyz):
    """Centroid shift + FPFH+RANSAC + ICP, all rigid (no scaling anywhere)."""
    pred = pred_xyz - pred_xyz.mean(0) + gt_xyz.mean(0)
    voxel = max(estimate_diam(gt_xyz) / 50.0, 0.02)
    pp, pf = fpfh(to_pcd(pred), voxel)
    gp, gf = fpfh(to_pcd(gt_xyz), voxel)
    dt = voxel * 1.5
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pp, gp, pf, gf, mutual_filter=True, max_correspondence_distance=dt,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4, checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dt)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    pred_pcd = to_pcd(pred)
    icp = o3d.pipelines.registration.registration_icp(
        pred_pcd, to_pcd(gt_xyz), dt * 2, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    pred_pcd.transform(icp.transformation)
    return np.asarray(pred_pcd.points), icp.fitness


def main():
    gt_map = {}
    for ln in open(GT_LIST):
        ln = ln.strip()
        if ln:
            gt_map[os.path.basename(ln).replace("_pointcloud.ply", "")] = ln

    per_arm = {a: {} for a in ARMS}
    for arm in ARMS:
        for npy in sorted(glob.glob(f"{OUT}/{arm}_*.npy")):
            oid = os.path.basename(npy)[len(arm) + 1:-4]
            if oid not in gt_map:
                continue
            xyz = np.load(npy)
            gt = load_ply_xyz(gt_map[oid])
            aligned, fit = align_rigid(xyz, gt)
            cds, c_p2g, _ = chamfer_sym(aligned, gt)
            per_arm[arm][oid] = {"cd_sym": float(cds), "cd_pred2gt": float(c_p2g), "icp_fit": float(fit)}
            print(f"{arm}/{oid}: cd_sym {cds:.4f} fit {fit:.2f}")

    common = sorted(set.intersection(*[set(per_arm[a]) for a in ARMS]))
    summary = {"n_common": len(common)}
    for a in ARMS:
        v = np.array([per_arm[a][o]["cd_sym"] for o in common])
        summary[a] = {"cd_sym_mean": float(v.mean()), "cd_sym_median": float(np.median(v))}
    for a in ARMS[1:]:
        d = np.array([per_arm[a][o]["cd_sym"] - per_arm["b0_control"][o]["cd_sym"] for o in common])
        summary[f"delta_{a}"] = {"paired_mean": float(d.mean()),
                                 "paired_sem": float(d.std(ddof=1) / np.sqrt(len(d))),
                                 "worse_count": int((d > 0).sum())}
    print(json.dumps(summary, indent=1))
    json.dump({"per_arm": per_arm, "summary": summary},
              open(f"{OUT}/chamfer_loo_noscale_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
