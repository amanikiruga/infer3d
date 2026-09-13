"""
Same direct-CD pipeline as eval_chamfer_direct.py but evaluates the finetuned
RGB+Depth+DINO Splatter Image baseline against the same FastGS pseudo-GT.
Reads PLYs from gs2mesh/input/realcars_dino_baseline_plys/ and writes
results/chamfer_dino_baseline.{csv,md}.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

from infer3d import config as _cfg
ROOT = _cfg.SPLATTER_REPO_ROOT
EVAL_DIR = f"{ROOT}/experiments/neurips_submission/real_ood_realcars_eval"
sys.path.append(EVAL_DIR)
from eval_chamfer_direct import (
    load_xyz, scene_dir_for_idx, car_center_from_cameras,
    crop_ball, align, chamfer_sym,
)

GS2MESH_INPUT = _cfg.GS2MESH_INPUT
OUT_DIR = Path(f"{EVAL_DIR}/results")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(20):
        idx = f"{i:02d}"
        pred_p = f"{GS2MESH_INPUT}/realcars_dino_baseline_plys/{idx}.ply"
        gt_p = f"{GS2MESH_INPUT}/realcars_pseudo_gt_plys/{idx}.ply"
        if not (os.path.exists(pred_p) and os.path.exists(gt_p)):
            print(f"[{idx}] missing ply, skip"); continue

        pred = load_xyz(pred_p)
        gt_full = load_xyz(gt_p)
        scene_dir = scene_dir_for_idx(idx)
        center = car_center_from_cameras(scene_dir)
        gt = crop_ball(gt_full, center, 3.0)
        if len(gt) < 1000:
            gt = crop_ball(gt_full, center, 4.5)

        pred_aln, fit, rmse = align(pred, gt)
        cd_sym, cd_p2g, cd_g2p = chamfer_sym(pred_aln, gt)
        print(f"[{idx}] n_pred={len(pred)} n_gt={len(gt)} CD p->g={cd_p2g:.5f} sym={cd_sym:.5f}")
        rows.append({
            "idx": idx, "n_pred": len(pred), "n_gt": len(gt),
            "cd_sym": cd_sym, "cd_p2g": cd_p2g, "cd_g2p": cd_g2p,
            "icp_fit": fit, "icp_rmse": rmse,
        })

    if not rows:
        print("no rows"); return

    csv_path = OUT_DIR / "chamfer_dino_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Pull existing ours/baseline/lgm/sf3d numbers from the 4col table CSV if present.
    p2g = np.array([r["cd_p2g"] for r in rows])
    sym = np.array([r["cd_sym"] for r in rows])

    md = OUT_DIR / "chamfer_dino_baseline.md"
    with open(md, "w") as f:
        f.write("# Splatter Image (RGB+Depth+DINO finetune) — Chamfer Distance vs FastGS pseudo-GT\n\n")
        f.write("Same protocol as eval_chamfer_direct.py: 3m sphere crop around camera-trajectory centroid, "
                "FPFH+RANSAC -> ICP, asymmetric pred->gt as the primary metric (single-view splat methods don't cover the back).\n\n")
        f.write("| idx | CD pred->gt | CD sym | n_pred | n_gt |\n")
        f.write("|-----|-------------|--------|--------|------|\n")
        for r in rows:
            f.write(f"| {r['idx']} | {r['cd_p2g']:.4f} | {r['cd_sym']:.4f} | {r['n_pred']} | {r['n_gt']} |\n")
        f.write(f"\n**Mean pred->gt:** {p2g.mean():.4f}  \n")
        f.write(f"**Median pred->gt:** {np.median(p2g):.4f}  \n")
        f.write(f"**Mean sym:** {sym.mean():.4f}  \n")
    print(f"saved {csv_path}\nsaved {md}")


if __name__ == "__main__":
    main()
