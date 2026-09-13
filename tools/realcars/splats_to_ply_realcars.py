"""
splats_to_ply_realcars.py
=========================

Stage 2 of the RealCars eval pipeline. For every idx 00..19 in
`experiments/neurips_submission/real_ood_realcars/<idx>`:
  - Load `inputs_cache/inputs.pth` (rgb_128 = the white-bg cropped GT input).
  - Run Splatter Image feedforward on rgb_128 -> baseline splats.
  - Load `final.pth["best"]["input"]` (the optimized DiffAE-decoded image).
  - Run Splatter Image feedforward on best_input, then bake `best.rotation` +
    `best.translation` into the splats via render_with_custom_camera_align(...,
    return_splats=True) -> ours splats.
  - Export both as GS-format PLYs.

Outputs:
  /.../gs2mesh/input/realcars_baseline_plys/<idx>.ply
  /.../gs2mesh/input/realcars_ours_plys/<idx>.ply

Pseudo-GT PLYs are already produced by FastGS at
  experiments/.../real_ood_realcars_eval/gs_pseudo_gt/<idx>/point_cloud/iteration_30000/point_cloud.ply
This script also symlinks them into
  /.../gs2mesh/input/realcars_pseudo_gt_plys/<idx>.ply
so all three method PLY folders sit side by side for Stage 3.

Usage:
  CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python -u \
    tools/realcars/splats_to_ply_realcars.py \
    abs=diffae_abs +dataset=cars \
    general.split=0 general.total_splits=1 \
    general.data_example_ids_path=not_needed.json \
    opt.pretrained_ckpt=<splatter-image SRN-Cars checkpoint>
"""
import os
import sys
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from plyfile import PlyData, PlyElement
from torch.utils.data import DataLoader

from infer3d import config as _cfg
ROOT = _cfg.SPLATTER_REPO_ROOT
GS2MESH_INPUT = Path(_cfg.GS2MESH_INPUT)
sys.path.append(ROOT)

from utils.abs_utils import render_with_custom_camera_align
from scene.gaussian_predictor import GaussianSplatPredictor
from splatter_image_datasets.srn import SRNDataset

device = torch.device("cuda")

OURS_OPT_ROOT = Path(f"{ROOT}/experiments/neurips_submission/real_ood_realcars")
PSEUDO_GT_ROOT = Path(f"{ROOT}/experiments/neurips_submission/real_ood_realcars_eval/gs_pseudo_gt")
OUT_OURS = GS2MESH_INPUT / "realcars_ours_plys"
OUT_BASELINE = GS2MESH_INPUT / "realcars_baseline_plys"
OUT_PSEUDO_GT = GS2MESH_INPUT / "realcars_pseudo_gt_plys"


def construct_list_of_attributes(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(features_dc_shape[1] * features_dc_shape[2]):
        l.append(f'f_dc_{i}')
    for i in range(features_rest_shape[1] * features_rest_shape[2]):
        l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(scaling_shape[1]):
        l.append(f'scale_{i}')
    for i in range(rotation_shape[1]):
        l.append(f'rot_{i}')
    return l


def export_splats_to_ply(splats, output_path, target_sh_degree=3):
    """Export splats dict to GS-format PLY (raw values: opacity pre-sigmoid, scale pre-exp, rot normalized)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = splats['xyz'].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    features_dc = splats['features_dc'].detach().cpu()
    features_rest = splats['features_rest'].detach().cpu()
    opacity = splats['opacity'].detach().cpu().numpy()
    scaling = splats['scaling'].detach().cpu().numpy()
    rotation = splats['rotation'].detach().cpu().numpy()

    # gp output already returns activated values (post-sigmoid opacity, post-exp scaling).
    # GS PLY format wants raw, so invert.
    opacity = -np.log((1.0 / (np.clip(opacity, 1e-7, 1 - 1e-7))) - 1.0)
    scaling = np.log(np.clip(scaling, 1e-8, None))

    features_dc = features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()
    features_rest = features_rest.transpose(1, 2).flatten(start_dim=1).contiguous()

    target_rest_features = 3 * (target_sh_degree + 1) ** 2 - 3
    cur = features_rest.shape[1]
    if cur < target_rest_features:
        features_rest = torch.cat([features_rest, torch.zeros(features_rest.shape[0], target_rest_features - cur)], dim=1)
    elif cur > target_rest_features:
        features_rest = features_rest[:, :target_rest_features]

    features_dc = features_dc.numpy()
    features_rest_np = features_rest.numpy()

    dc_shape = (features_dc.shape[0], 1, features_dc.shape[1])
    rest_shape = (features_rest_np.shape[0], 1, features_rest_np.shape[1])
    dtype_full = [(a, 'f4') for a in construct_list_of_attributes(
        dc_shape, rest_shape, splats['scaling'].shape, splats['rotation'].shape)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, features_dc, features_rest_np, opacity, scaling, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(str(output_path))
    print(f"  -> {output_path}  ({xyz.shape[0]} gaussians)")


@hydra.main(version_base=None, config_path=f"{ROOT}/configs", config_name="abs_config")
def main(cfg: DictConfig):
    OUT_OURS.mkdir(parents=True, exist_ok=True)
    OUT_BASELINE.mkdir(parents=True, exist_ok=True)
    OUT_PSEUDO_GT.mkdir(parents=True, exist_ok=True)

    # gp + canonical source-view transforms (from one SRN-cars train sample)
    gp = GaussianSplatPredictor(cfg).to(memory_format=torch.channels_last).to(device).eval()
    ck = torch.load(cfg.opt.pretrained_ckpt, map_location=device, weights_only=False)
    gp.load_state_dict(ck["model_state_dict"])
    print(f"[stage2] loaded gp from {cfg.opt.pretrained_ckpt}")

    train_ds = SRNDataset(cfg, "train", data_category="cars")
    src = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))
    v2w = src["view_to_world_transforms"][:, :cfg.data.input_images, ...].to(device)
    cv2wT = src["source_cv2wT_quat"][:, :cfg.data.input_images].to(device)
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)
    zgt = 1.3

    idxs = sorted([p.name for p in OURS_OPT_ROOT.iterdir() if p.is_dir() and p.name.isdigit()])
    print(f"[stage2] found {len(idxs)} optim runs")

    for idx in idxs:
        sub = OURS_OPT_ROOT / idx
        cache = sub / "inputs_cache" / "inputs.pth"
        final = sub / "final.pth"
        if not cache.exists() or not final.exists():
            print(f"[{idx}] skip — incomplete optim outputs"); continue

        bundle = torch.load(cache, map_location=device, weights_only=False)
        rgb_white = bundle["rgb_128"].to(device)               # [1,3,128,128]
        fin = torch.load(final, map_location=device, weights_only=False)
        best_input = fin["best"]["input"].to(device).unsqueeze(0)  # [1,3,128,128]
        best_R = fin["best"]["rotation"].to(device)             # [3,3]
        best_T = fin["best"]["translation"].to(device)          # [3]

        with torch.no_grad():
            # Baseline = vanilla Splatter Image on white-bg input
            splats_b = gp(rgb_white.unsqueeze(1), v2w, cv2wT, None)
            splats_b = {k: v[0] for k, v in splats_b.items()}

            # Ours = Splatter Image on optimized DiffAE input, then apply optimized rigid pose
            splats_o = gp(best_input.unsqueeze(1), v2w, cv2wT, None)
            splats_o = {k: v[0] for k, v in splats_o.items()}
            splats_o = render_with_custom_camera_align(
                splats_o, bg, cfg, None, best_R, zgt, device=device,
                return_splats=True, translation=best_T, zgt_ood=zgt,
            )

        print(f"[{idx}] exporting baseline + ours")
        export_splats_to_ply(splats_b, OUT_BASELINE / f"{idx}.ply")
        export_splats_to_ply(splats_o, OUT_OURS / f"{idx}.ply")

        # Pseudo-GT symlink (FastGS produces a normal GS PLY)
        pgt_src = PSEUDO_GT_ROOT / idx / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        pgt_dst = OUT_PSEUDO_GT / f"{idx}.ply"
        if pgt_src.exists():
            if pgt_dst.exists() or pgt_dst.is_symlink():
                pgt_dst.unlink()
            pgt_dst.symlink_to(pgt_src)
            print(f"  -> pseudo_gt symlink {pgt_dst}")
        else:
            print(f"  !! pseudo_gt missing for {idx}: {pgt_src}")

    print("[stage2] DONE")


if __name__ == "__main__":
    main()
