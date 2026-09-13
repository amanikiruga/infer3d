"""Gauge/scale-free ICP-aligned Chamfer scorer for Task-A geometry.

Aligns each method's predicted point cloud to the GT ShapeNet surface (pre-scale to
GT diameter -> FPFH+RANSAC -> ICP), then symmetric Chamfer. Identical alignment for
every method => fair shape metric, invariant to each method's canonical frame.
(align/chamfer_sym copied verbatim from the paper's realcars eval_chamfer_direct.py.)

  python icp_chamfer.py --pred_dir <dir of {obj}.npy> --tag <name>
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, csv, json, os
import numpy as np, open3d as o3d

WT = f"{_EXT}/splatter-image-rebuttal"
GT = f"{WT}/rebuttal/results/task_a_geom/gt"


def estimate_diam(xyz):
    if len(xyz) == 0:
        return 1.0
    return float(np.linalg.norm(xyz.max(0) - xyz.min(0)))


def to_pcd(xyz):
    p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(xyz); return p


def fpfh(pcd, voxel):
    d = pcd.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return d, f


def align(pred_xyz, gt_xyz):
    gt_diam = estimate_diam(gt_xyz)
    scale = gt_diam / max(estimate_diam(pred_xyz), 1e-6)
    pred = (pred_xyz - pred_xyz.mean(0)) * scale + gt_xyz.mean(0)
    voxel = max(gt_diam / 50.0, 0.02)
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


def align_transform(pred_xyz, gt_xyz):
    """Same global registration as align(), but RETURN the composed similarity
    (s, R, t) with  X_gt ~= s * R @ X_pred + t  (pred in its own frame -> GT/NMR frame).
    Used by the oracle-pose NVS to map GT cameras into each method's frame."""
    gt_diam = estimate_diam(gt_xyz)
    scale = gt_diam / max(estimate_diam(pred_xyz), 1e-6)
    pred_mean = pred_xyz.mean(0); gt_mean = gt_xyz.mean(0)
    pred = (pred_xyz - pred_mean) * scale + gt_mean
    voxel = max(gt_diam / 50.0, 0.02)
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
    icp = o3d.pipelines.registration.registration_icp(
        to_pcd(pred), to_pcd(gt_xyz), dt * 2, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    T = np.asarray(icp.transformation)          # maps `pred` (pre-scaled) -> GT
    Rt, tt = T[:3, :3], T[:3, 3]
    # compose:  X_gt = Rt @ (scale*(X_pred - pred_mean) + gt_mean) + tt
    s = float(scale)
    R = Rt
    t = Rt @ (gt_mean - scale * pred_mean) + tt
    # verify
    resid = (s * (R @ pred_xyz.T).T + t) - (np.asarray(to_pcd(pred).transform(T).points))
    return s, R, t, float(icp.fitness), float(np.abs(resid).max())


def chamfer_sym(a, b):
    pa, pb = to_pcd(a), to_pcd(b)
    d_ab = np.asarray(pa.compute_point_cloud_distance(pb))
    d_ba = np.asarray(pb.compute_point_cloud_distance(pa))
    ca, cb = float(np.mean(d_ab ** 2)), float(np.mean(d_ba ** 2))
    return 0.5 * (ca + cb), ca, cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_geom")
    args = ap.parse_args()
    rows = []
    for f in sorted(os.listdir(GT)):
        oid = f[:-4]
        pp = f"{args.pred_dir}/{oid}.npy"
        if not os.path.exists(pp):
            continue
        gt = np.load(f"{GT}/{oid}.npy"); pred = np.load(pp)
        try:
            aln, fit = align(pred, gt)
            cds, ca, cb = chamfer_sym(aln, gt)
        except Exception as e:
            print(f"{oid}: align/cd FAIL {e}"); continue
        rows.append({"obj": oid, "cd_sym": cds, "cd_pred2gt": ca, "cd_gt2pred": cb, "icp_fit": fit})
        print(f"{oid}: CD_sym={cds:.4f} pred2gt={ca:.4f} fit={fit:.2f}")
    if not rows:
        print("no rows"); return
    cds = np.array([r["cd_sym"] for r in rows])
    p2g = np.array([r["cd_pred2gt"] for r in rows])
    out_csv = f"{args.out}/chamfer_{args.tag}.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = {"tag": args.tag, "n": len(rows),
            "cd_sym_mean": float(cds.mean()), "cd_sym_median": float(np.median(cds)),
            "cd_pred2gt_mean": float(p2g.mean()), "cd_pred2gt_median": float(np.median(p2g))}
    json.dump(summ, open(f"{args.out}/chamfer_{args.tag}_summary.json", "w"), indent=1)
    print(f"\n[{args.tag}] n={len(rows)} CD_sym mean={cds.mean():.4f} median={np.median(cds):.4f} "
          f"| pred2gt mean={p2g.mean():.4f}\n-> {out_csv}")


if __name__ == "__main__":
    main()
