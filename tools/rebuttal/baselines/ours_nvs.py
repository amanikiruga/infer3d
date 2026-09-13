"""Render OURS at the exported Task-A target cameras (same order as NFI/GT) and
score — validates ours' NVS numbers on the shared harness and produces aligned
frames for the comparison video. Reproduces ours' eval_pred splat transform
(best_input_image -> lifter -> apply best_rotation_matrix)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, glob, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, WT)
os.environ.setdefault("SPLATTER_IMAGE_ROOT", WT)
os.environ.setdefault("SHAPENET_NMR_ROOT", f"{_EXT}/datasets/srn_nmr_classes")
from hydra import compose, initialize_config_dir
from scene.gaussian_predictor import GaussianSplatPredictor
from utils.abs_utils import render_with_custom_camera
sys.path.insert(0, f"{WT}/rebuttal/baselines")
from scorer import render_relative, score_views, bundle_targets
dev = "cuda"
LIFTER = f"{_EXT}/splatter-image/experiments_out/2025-08-06/12-11-04/model_latest.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{WT}/rebuttal/results/task_a_runs/ours")
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_runs/ours_nvs")
    args = ap.parse_args()
    with initialize_config_dir(version_base=None, config_dir=f"{WT}/configs"):
        cfg = compose(config_name="abs_config", overrides=["+dataset=shapenet-nmr"])
    gp = GaussianSplatPredictor(cfg).to(dev).eval()
    gp.load_state_dict(torch.load(LIFTER, map_location=dev, weights_only=False)["model_state_dict"])
    v2w = torch.eye(4, device=dev).reshape(1, 1, 4, 4)
    quat = torch.tensor([1., 0, 0, 0], device=dev).reshape(1, 1, 4)
    bg = torch.tensor([1., 1, 1], device=dev)
    rows = []
    for d in sorted(glob.glob(f"{args.runs}/*/topk_best_everything_latest.pth")):
        oid = os.path.basename(os.path.dirname(d)).split("-")[0]
        bdir = f"{args.bundles}/{oid}"
        if not os.path.isdir(bdir):
            continue
        c = torch.load(d, map_location=dev, weights_only=False)
        img = c["best_input_image"].to(dev)
        if img.dim() == 4:
            img = img.unsqueeze(1)
        rot = c["best_rotation_matrix"].to(dev)
        zgt = float(c.get("zgt", 1.3))
        with torch.no_grad():
            sp = gp(img, v2w, quat, None)
            sp = {k: v[0] for k, v in sp.items()}
            tsp = render_with_custom_camera(sp, bg, cfg, None, rot, zgt, device=dev, return_splats=True)
        odir = f"{args.out}/{oid}"
        outs = render_relative(tsp, bdir, odir, cfg)
        m = score_views(outs, bundle_targets(bdir)); m["object"] = oid
        rows.append(m)
        print(f"{oid}: PSNR {m['PSNR']:.2f} SSIM {m['SSIM']:.3f} LPIPS {m['LPIPS']:.3f}")
    if rows:
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ["PSNR", "SSIM", "LPIPS"]}
        print(f"\nOURS NVS @ exported cams (n={len(rows)}): {mean}")
        import json; json.dump({"rows": rows, "mean": mean}, open(f"{args.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
