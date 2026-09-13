"""CO3D in-distribution render pair (Infer3D vs Splatter) for the frequency analysis.

Mirrors experiments/stylegan3-ours-co3d/eval_renderings_with_icp_batch.evaluate_checkpoint
(the code that produced the paper Table-1 numbers) but, instead of only aggregating
metrics, saves the ICP-aligned per-view float renders as lossless PNG stacks for both
methods plus the GT views, so the CPU frequency_gap.py can decompose the ID gap.

Both methods are ICP-aligned to the pseudo-GT point cloud (identical treatment), exactly
as in Table 1. Cached optimization checkpoints + eval PLYs live in the ORIGINAL
splatter-image repo; outputs are written into the rebuttal worktree.

Usage (DiffAE hydrants ID, GPU 0):
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled mamba run -n test2 python render_id_pair_co3d.py \
      --ckpt_dir  checkpoints-diffae-co3d-hydrants-baseline-se3-new-stylegan-10-depth-encoder-indist \
      --dataset hydrants --lifter hydrants --tag hyd_diffae_id [--limit N]
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
from PIL import Image

WT = f"{_EXT}/splatter-image-rebuttal"
ORIG = f"{_EXT}/splatter-image"
sys.path.insert(0, WT)                                   # worktree modules win
sys.path.insert(0, f"{WT}/experiments/stylegan3-ours-co3d")
import eval_renderings_with_icp_batch as E               # noqa: E402
from hydra import initialize_config_dir, compose         # noqa: E402
import hydra                                             # noqa: E402

LIFTERS = {
    "hydrants": f"{ORIG}/experiments_out/2025-10-07/16-13-13/model_latest.pth",
    "vases":    f"{ORIG}/experiments_out/2026-01-15/14-55-34/model_latest.pth",
}
RES = f"{WT}/rebuttal/results/freq_co3d"

from utils.abs_utils import render_with_custom_camera  # noqa: E402


def regenerate_ours_splats_paper(checkpoint, gp, cfg, device):
    """Verbatim `regenerate_ours_splats` from commit e534f0e ("have all numbers ...",
    2026-01-04) — the code that produced the cached CSVs / paper Table-1 ID cells. The
    current worktree version diverged (render_with_custom_camera_align + zgt_ood + a
    different view_to_world slice), under-reproducing ours by ~2-4 dB. We analyze the
    renders behind the *paper's* numbers, so we reproduce this exact version. Splatter
    (regenerate_baseline_ood_splats) is unchanged and matches the cache byte-for-byte."""
    best_input_image = checkpoint["best_input_image"]
    best_rotation_matrix = checkpoint["best_rotation_matrix"].to(device)
    best_translation_matrix = checkpoint["best_translation_matrix"].to(device)
    zgt = checkpoint["zgt"]
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in checkpoint["ood_data"].items()}
    ood_image_index = checkpoint["ood_image_index"]
    ood_random_rotation = checkpoint.get("ood_random_rotation", None)
    if ood_random_rotation is None:
        raise ValueError("ood_random_rotation not found in checkpoint")
    ood_random_rotation = ood_random_rotation.to(device)
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    input_image = best_input_image
    if input_image.dim() == 4:
        input_image = input_image.squeeze(0)
    if input_image.shape[0] > 3:
        input_image = input_image[:3, :, :]
    input_images = torch.cat([input_image.unsqueeze(0).unsqueeze(1).to(device),
                              ood_origin_distances.unsqueeze(0).unsqueeze(1)], dim=2)
    focals_pixels_pred = ood_focals_pixels.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred_splats = gp(
            input_images,
            ood_data["view_to_world_transforms"][:1, ood_image_index:ood_image_index + cfg.data.input_images],
            ood_data["source_cv2wT_quat"][:1, ood_image_index:ood_image_index + cfg.data.input_images],
            focals_pixels_pred)
    pred_splats = {k: v[0] for k, v in pred_splats.items()}
    transformed = render_with_custom_camera(pred_splats, bg, cfg, ood_focals_pixels,
                                            best_rotation_matrix, zgt, device=device,
                                            return_splats=True, translation=best_translation_matrix)
    inverse_rotation = ood_random_rotation.T
    transformed = render_with_custom_camera(transformed, bg, cfg, ood_focals_pixels,
                                            inverse_rotation, zgt, device=device,
                                            return_splats=True, translation=None)
    return transformed, ood_data


def save_stack(pairs, method_dir, gt_dir, oid):
    os.makedirs(f"{method_dir}/{oid}", exist_ok=True)
    os.makedirs(f"{gt_dir}/{oid}/targets", exist_ok=True)
    for i, (pred, gt) in enumerate(pairs):
        Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8)).save(f"{method_dir}/{oid}/{i:02d}.png")
        Image.fromarray((np.clip(gt, 0, 1) * 255).astype(np.uint8)).save(f"{gt_dir}/{oid}/targets/{i:02d}.png")


def render_views(splats, ood_data, cfg, device, ood_image_index):
    if ood_image_index != 0:
        from copy import deepcopy
        rel = E.make_data_relative_to_idx(deepcopy(ood_data), ood_image_index)
    else:
        rel = ood_data
    rel = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in rel.items()}
    splats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in splats.items()}
    bg = torch.tensor([1., 1, 1] if cfg.data.white_background else [0., 0, 0], device=device)
    n = rel["world_view_transforms"].shape[1]
    out = []
    for r in range(n):
        wvt = rel["world_view_transforms"][0, r].unsqueeze(0)
        fpt = rel["full_proj_transforms"][0, r].unsqueeze(0)
        cc = rel["camera_centers"][0, r].unsqueeze(0)
        foc = rel["focals_pixels"][0, r] if "focals_pixels" in rel else None
        with torch.no_grad():
            img = E.render_predicted(splats, wvt, fpt, cc, bg, cfg, focals_pixels=foc)["render"].clamp(0, 1)
        gt = rel["gt_images"][0, r].to(device).clamp(0, 1)
        out.append((img.permute(1, 2, 0).cpu().numpy().astype(np.float32),
                    gt.permute(1, 2, 0).cpu().numpy().astype(np.float32)))
    return out


def icp_align(pred_ply, gt_ply, device, num_points=100000, max_iter=100):
    pp = E.load_pointcloud(pred_ply, num_points)
    gp = E.load_pointcloud(gt_ply, num_points)
    off = np.mean(gp, 0) - np.mean(pp, 0)
    X = torch.from_numpy(pp + off).float().unsqueeze(0).to(device)
    Y = torch.from_numpy(gp).float().unsqueeze(0).to(device)
    sol = E.iterative_closest_point(X, Y, estimate_scale=True, max_iterations=max_iter, verbose=False)
    return sol, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="relative to original splatter-image repo")
    ap.add_argument("--dataset", default="hydrants")
    ap.add_argument("--lifter", default="hydrants")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda"
    ckpt_dir = f"{ORIG}/{args.ckpt_dir}"
    eval_dir = f"{ORIG}/eval_output/{args.ckpt_dir}"

    # Faithful ICP: use the exact point clouds the paper eval pipeline consumed,
    # listed (one path per object) in lists/{ours,baseline_ood,pseudo_gt}.txt.
    def ply_map(name):
        d = {}
        for ln in open(f"{eval_dir}/lists/{name}.txt"):
            ln = ln.strip()
            if not ln:
                continue
            oid = os.path.basename(ln).replace("_pointcloud.ply", "")
            d[oid] = ln
        return d
    ours_map, base_map, gt_map = ply_map("ours"), ply_map("baseline_ood"), ply_map("pseudo_gt")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{WT}/configs")
    cfg = compose(config_name="abs_config", overrides=[
        "general.split=0", "general.total_splits=1", f"+dataset={args.dataset}",
        "general.prefix=freq-co3d", "abs=stylegan_abs",
        f"opt.pretrained_ckpt={LIFTERS[args.lifter]}",
        "general.data_example_ids_path=not_needed.json"])
    gp = E.load_gaussian_predictor(cfg, device)

    ours_dir = f"{RES}/{args.tag}/ours"
    splat_dir = f"{RES}/{args.tag}/splatter"
    gt_dir = f"{RES}/{args.tag}/gt"
    val_dataset = None
    ours_rows, splat_rows = [], []

    oids = sorted(set(ours_map) & set(base_map) & set(gt_map))
    if args.limit:
        oids = oids[:args.limit]
    for oid in oids:
        ck = glob.glob(f"{ckpt_dir}/{oid}*/topk_best_everything_latest.pth")
        op = [ours_map[oid]]; bp = [base_map[oid]]; gtp = [gt_map[oid]]
        if not (ck and os.path.exists(op[0]) and os.path.exists(bp[0]) and os.path.exists(gtp[0])):
            print(f"  skip {oid} (missing)"); continue
        try:
            checkpoint = E.load_checkpoint(ck[0], device)
            if "ood_data" not in checkpoint:
                if val_dataset is None:
                    val_dataset = E.CO3DDataset(cfg, "test", use_hq=False)
                idx, _ = val_dataset.find_index_for_sequence_prefix(checkpoint["example_id"])
                od = val_dataset[idx]
                od = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in od.items()}
                checkpoint["ood_data"] = od
            oii = checkpoint.get("ood_image_index", 0)
            gt_splats = E.regenerate_baseline_indist_splats(checkpoint, gp, cfg, device)

            # ours (ICP-aligned)
            sol_o, off_o = icp_align(op[0], gtp[0], device)
            pred_o, ood_data = regenerate_ours_splats_paper(checkpoint, gp, cfg, device)
            pred_o_icp = E.apply_icp_to_splats(pred_o, sol_o, off_o, device)
            pairs_o = render_views(pred_o_icp, ood_data, cfg, device, oii)
            save_stack(pairs_o, ours_dir, gt_dir, oid)

            # splatter baseline (ICP-aligned)
            sol_b, off_b = icp_align(bp[0], gtp[0], device)
            pred_b, ood_data2 = E.regenerate_baseline_ood_splats(checkpoint, gt_splats, gp, cfg, device)
            pred_b_icp = E.apply_icp_to_splats(pred_b, sol_b, off_b, device)
            pairs_b = render_views(pred_b_icp, ood_data2, cfg, device, oii)
            save_stack(pairs_b, splat_dir, gt_dir, oid)  # gt identical; harmless re-save

            po = float(np.mean([-10*np.log10(np.mean((p-g)**2)+1e-12) for p, g in pairs_o]))
            pb = float(np.mean([-10*np.log10(np.mean((p-g)**2)+1e-12) for p, g in pairs_b]))
            ours_rows.append(po); splat_rows.append(pb)
            print(f"{oid}: ours {po:.2f} | splatter {pb:.2f}")
        except Exception as ex:
            import traceback; traceback.print_exc(); print(f"  ERROR {oid}: {ex}")

    print(f"\nOURS   n={len(ours_rows)} meanPSNR {np.mean(ours_rows):.3f}")
    print(f"SPLAT  n={len(splat_rows)} meanPSNR {np.mean(splat_rows):.3f}")
    os.makedirs(f"{RES}/{args.tag}", exist_ok=True)
    json.dump({"ours_psnr": ours_rows, "splat_psnr": splat_rows,
               "ours_mean": float(np.mean(ours_rows)), "splat_mean": float(np.mean(splat_rows))},
              open(f"{RES}/{args.tag}/render_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
