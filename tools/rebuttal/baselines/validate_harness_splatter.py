"""Harness validation: Splatter-Image lifter on exported Task-A bundles.

Runs the frozen NMR lifter on each bundle's input.png (identity input pose, as in
eval_baseline) and renders the exported relative target cams. The resulting mean
PSNR must match the probe's cars 'relative baseline' (~20.6, n=22) — proving the
exported cameras/images/metric are byte-equivalent to the paper pipeline.

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled mamba run -n test2 python validate_harness_splatter.py
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json
import os
import sys

import numpy as np
import torch

WORKTREE = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, WORKTREE)
os.environ.setdefault("SPLATTER_IMAGE_ROOT", WORKTREE)
os.environ.setdefault(
    "SHAPENET_NMR_ROOT",
    f"{_EXT}/datasets/srn_nmr_classes")

from hydra import compose, initialize_config_dir

from scene.gaussian_predictor import GaussianSplatPredictor
from scorer import score_views, render_relative, bundle_targets, load_png

LIFTER = (f"{_EXT}/splatter-image/"
          "experiments_out/2025-08-06/12-11-04/model_latest.pth")
RES = f"{WORKTREE}/rebuttal/results"
device = "cuda"


def main():
    with initialize_config_dir(version_base=None, config_dir=f"{WORKTREE}/configs"):
        cfg = compose(config_name="abs_config", overrides=["+dataset=shapenet-nmr"])
    gp = GaussianSplatPredictor(cfg).to(device).eval()
    ck = torch.load(LIFTER, map_location=device, weights_only=False)
    gp.load_state_dict(ck["model_state_dict"])

    v2w = torch.eye(4, device=device).reshape(1, 1, 4, 4)
    quat = torch.tensor([1., 0, 0, 0], device=device).reshape(1, 1, 4)

    ids = json.load(open(f"{RES}/task_a_car_ids.json"))
    rows = []
    for oid in ids:
        bdir = f"{RES}/task_a_inputs/{oid}"
        img = load_png(f"{bdir}/input.png").to(device)
        with torch.no_grad():
            splats = gp(img.unsqueeze(0).unsqueeze(1), v2w, quat, None)
        splats = {k: v[0] for k, v in splats.items()}
        outs = render_relative(splats, bdir,
                               f"{RES}/task_a_runs/splatter_relative/{oid}", cfg)
        m = score_views(outs, bundle_targets(bdir))
        m["object"] = oid
        rows.append(m)
        print(f"{oid}: PSNR {m['PSNR']:.2f} SSIM {m['SSIM']:.3f} LPIPS {m['LPIPS']:.3f}")

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ["PSNR", "SSIM", "LPIPS"]}
    print(f"\nVALIDATION MEAN (n={len(rows)}): {mean}")
    json.dump({"rows": rows, "mean": mean},
              open(f"{RES}/task_a_runs/splatter_relative/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
