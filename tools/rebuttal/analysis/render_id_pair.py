"""Render Infer3D vs Splatter-Image on the SAME in-distribution (control) bundle.

For the frequency-gap analysis (rebuttal, reviewer XsZX / AC priority 2) we need,
per object at the in-distribution pose, three view-aligned PNG stacks:
  - Infer3D (ours)  : optimized generate-then-lift, from ours_control checkpoint
  - Splatter Image  : frozen feed-forward lifter on the raw input.png
  - GT targets       : the bundle's 24-view test ring
Both methods are rendered with the SAME bundle cameras and scored with the SAME
metric code (scorer.score_views) so the comparison is byte-consistent with the
paper pipeline.  GPU cost is tiny: one lifter forward per object + 24 renders x2.

Validation targets (must hold, else the bundle choice is wrong):
  ours  mean PSNR ~= 21.4  (paper Table-2 In-dist. Infer3D(SG) 21.61)
  splat mean PSNR ~= 24    (paper Table-2 In-dist. Splatter Img. 24.27)

Usage:
  CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled mamba run -n test2 \
    python render_id_pair.py --bundles task_a_frontal
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, WT)
os.environ.setdefault("SPLATTER_IMAGE_ROOT", WT)
os.environ.setdefault(
    "SHAPENET_NMR_ROOT",
    f"{_EXT}/datasets/srn_nmr_classes")
sys.path.insert(0, f"{WT}/rebuttal/baselines")

from hydra import compose, initialize_config_dir
from scene.gaussian_predictor import GaussianSplatPredictor
from utils.abs_utils import render_with_custom_camera
from scorer import render_relative, score_views, bundle_targets, load_png

DEV = "cuda"
RES = f"{WT}/rebuttal/results"
LIFTER = (f"{_EXT}/splatter-image/"
          "experiments_out/2025-08-06/12-11-04/model_latest.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default="task_a_frontal",
                    choices=["task_a_frontal", "task_a_inputs_control"])
    ap.add_argument("--ours_runs", default=f"{RES}/task_a_runs/ours_control")
    ap.add_argument("--out_ours", default=f"{RES}/task_a_runs/freq_ours_control")
    ap.add_argument("--out_splat", default=f"{RES}/task_a_runs/freq_splatter_control")
    args = ap.parse_args()
    bundles = f"{RES}/{args.bundles}"

    with initialize_config_dir(version_base=None, config_dir=f"{WT}/configs"):
        cfg = compose(config_name="abs_config", overrides=["+dataset=shapenet-nmr"])
    gp = GaussianSplatPredictor(cfg).to(DEV).eval()
    gp.load_state_dict(torch.load(LIFTER, map_location=DEV, weights_only=False)["model_state_dict"])
    v2w = torch.eye(4, device=DEV).reshape(1, 1, 4, 4)
    quat = torch.tensor([1., 0, 0, 0], device=DEV).reshape(1, 1, 4)
    bg = torch.tensor([1., 1, 1], device=DEV)

    ours_rows, splat_rows = [], []
    ckpts = sorted(glob.glob(f"{args.ours_runs}/*/topk_best_everything_latest.pth"))
    for ck in ckpts:
        oid = os.path.basename(os.path.dirname(ck)).split("-")[0]
        bdir = f"{bundles}/{oid}"
        if not os.path.isdir(bdir):
            continue
        targets = bundle_targets(bdir)

        # --- Infer3D (ours): generate-then-lift, at optimized rotation ---
        c = torch.load(ck, map_location=DEV, weights_only=False)
        img = c["best_input_image"].to(DEV)
        if img.dim() == 4:
            img = img.unsqueeze(1)
        rot = c["best_rotation_matrix"].to(DEV)
        zgt = float(c.get("zgt", 1.3))
        with torch.no_grad():
            sp = gp(img, v2w, quat, None)
            sp = {k: v[0] for k, v in sp.items()}
            tsp = render_with_custom_camera(sp, bg, cfg, None, rot, zgt,
                                            device=DEV, return_splats=True)
        o_out = render_relative(tsp, bdir, f"{args.out_ours}/{oid}", cfg)
        mo = score_views(o_out, targets); mo["object"] = oid
        ours_rows.append(mo)

        # --- Splatter Image: frozen feed-forward on the raw input ---
        raw = load_png(f"{bdir}/input.png").to(DEV)
        with torch.no_grad():
            spb = gp(raw.unsqueeze(0).unsqueeze(1), v2w, quat, None)
            spb = {k: v[0] for k, v in spb.items()}
        s_out = render_relative(spb, bdir, f"{args.out_splat}/{oid}", cfg)
        ms = score_views(s_out, targets); ms["object"] = oid
        splat_rows.append(ms)
        print(f"{oid}: ours {mo['PSNR']:.2f} | splatter {ms['PSNR']:.2f}")

    def summ(rows):
        return {k: float(np.mean([r[k] for r in rows]))
                for k in ["PSNR", "SSIM", "LPIPS"]} | {"n": len(rows)}
    so, ss = summ(ours_rows), summ(splat_rows)
    print(f"\nOURS   (n={so['n']}): PSNR {so['PSNR']:.3f} SSIM {so['SSIM']:.4f} LPIPS {so['LPIPS']:.4f}"
          f"   [paper ID 21.61]")
    print(f"SPLAT  (n={ss['n']}): PSNR {ss['PSNR']:.3f} SSIM {ss['SSIM']:.4f} LPIPS {ss['LPIPS']:.4f}"
          f"   [paper ID 24.27]")
    json.dump({"bundle": args.bundles, "ours": {"summary": so, "rows": ours_rows},
               "splatter": {"summary": ss, "rows": splat_rows}},
              open(f"{RES}/task_a_runs/freq_id_render_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
