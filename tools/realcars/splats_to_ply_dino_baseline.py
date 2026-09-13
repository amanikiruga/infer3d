"""
Stage 2 (extras variant): export PLYs from the finetuned RGB+Depth+DINO Splatter
Image baseline so we can run the same direct-CD eval against FastGS pseudo-GT.

Outputs:
  /.../gs2mesh/input/realcars_dino_baseline_plys/<idx>.ply   (20 of these)

Usage:
  CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python -u \
    experiments/neurips_submission/real_ood_realcars_eval/splats_to_ply_dino_baseline.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from infer3d import config as _cfg
ROOT = _cfg.SPLATTER_REPO_ROOT
sys.path.append(ROOT)
sys.path.append(f"{ROOT}/experiments/neurips_submission/real_ood_realcars_eval")

from torch.utils.data import DataLoader
from splatter_image_datasets.srn import SRNDataset
from experiments.neurips_submission.srn_cars_depth_dino_baseline.predictor_with_extras import (
    GaussianSplatPredictorExtras,
)
from splats_to_ply_realcars import export_splats_to_ply

device = torch.device("cuda")

OURS_OPT_ROOT = Path(f"{ROOT}/experiments/neurips_submission/real_ood_realcars")
OUT_DIR = Path(_cfg.GS2MESH_INPUT) / "realcars_dino_baseline_plys"

EXTRAS_CKPT = f"{ROOT}/experiments_out/2026-05-05/15-03-06/model_latest.pth"
EXTRAS_CFG = f"{ROOT}/experiments/neurips_submission/srn_cars_depth_dino_baseline/configs/train_extras.yaml"
PCA_BASIS = _cfg.DINO_PCA_BASIS
DINO_PCA_DIM = 32


def normalize_depth(depth):
    inv = 1.0 / depth.clamp(min=1e-3)
    V = inv.shape[0]
    flat = inv.view(V, -1)
    lo = flat.min(dim=1).values.view(V, 1, 1, 1)
    hi = flat.max(dim=1).values.view(V, 1, 1, 1)
    return (inv - lo) / (hi - lo + 1e-6)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(EXTRAS_CFG)

    gp = GaussianSplatPredictorExtras(cfg).to(memory_format=torch.channels_last).to(device).eval()
    ck = torch.load(EXTRAS_CKPT, map_location=device, weights_only=False)
    gp.load_state_dict(ck["model_state_dict"], strict=True)
    print(f"[stage2-dino] loaded extras gp from {EXTRAS_CKPT} (iter {ck.get('iteration','?')})")

    pca = torch.load(PCA_BASIS, map_location=device, weights_only=False)
    pca_mean = pca["mean"].to(device)
    pca_basis = pca["basis"][:, :DINO_PCA_DIM].to(device)

    train_ds = SRNDataset(cfg, "train", data_category="cars")
    src = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))
    v2w = src["view_to_world_transforms"][:, :cfg.data.input_images, ...].to(device)
    cv2wT = src["source_cv2wT_quat"][:, :cfg.data.input_images].to(device)

    idxs = sorted([p.name for p in OURS_OPT_ROOT.iterdir() if p.is_dir() and p.name.isdigit()])
    print(f"[stage2-dino] {len(idxs)} idxs")

    for idx in idxs:
        cache = OURS_OPT_ROOT / idx / "inputs_cache" / "inputs.pth"
        if not cache.exists():
            print(f"[{idx}] skip — incomplete"); continue

        b = torch.load(cache, map_location=device, weights_only=False)
        rgb_white = b["rgb_128"].to(device)
        depth_128 = b["da3_depth_128"].to(device).float()
        dino_grid = b["dino_feat_grid"].to(device).float()

        depth_norm = normalize_depth(depth_128)
        V, C, h, w = dino_grid.shape
        flat = dino_grid.permute(0, 2, 3, 1).reshape(-1, C) - pca_mean
        proj = (flat @ pca_basis).reshape(V, h, w, DINO_PCA_DIM).permute(0, 3, 1, 2).contiguous()
        dino_up = F.interpolate(proj, size=(128, 128), mode="bilinear", align_corners=False)
        x_extras = torch.cat([rgb_white, depth_norm, dino_up], dim=1).unsqueeze(1)

        with torch.no_grad():
            splats = gp(x_extras, v2w, cv2wT, None)
            splats = {k: v[0] for k, v in splats.items()}

        print(f"[{idx}] exporting dino_baseline")
        export_splats_to_ply(splats, OUT_DIR / f"{idx}.ply")

    print("[stage2-dino] DONE")


if __name__ == "__main__":
    main()
