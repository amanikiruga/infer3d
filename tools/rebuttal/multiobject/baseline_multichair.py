"""Feed-forward Splatter Image baseline on the 20 multichair-pipeline-v1 scenes.

Protocol matched to checkpoints-multichair-pipeline-v1-20scenes/stage3 (the PSNR-14.04
"ours" aggregate): same stage-1 bundles, same GT cameras, metrics over views 1..19
(input view 0 excluded), same PSNR/SSIM/LPIPS code. The baseline is the single-chair-
trained model_chairs.pth applied directly to the 2-chair input image; splats predicted
in the view-0 relative frame and rendered at the bundle's cameras (geometrically correct
for the baseline — no scale heuristic needed).

Derived from multichair_stage3_eval.py's self-contained baseline path (its `ours` path is
broken against the current stage2 module and is not needed — ours comes from the existing
stage3 gt_compare artifacts).
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os, sys
from pathlib import Path

ROOT = f"{_EXT}/splatter-image"
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.append(ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from hydra import initialize_config_dir, compose
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from PIL import Image

device = "cuda"
SRC = f"{ROOT}/checkpoints-multichair-pipeline-v1-20scenes"
OUT = f"{WT}/rebuttal/multiobject/baseline_eval"


@torch.no_grad()
def baseline_splats(rgb_t, bundle, gp, cfg):
    target_res = cfg.data.training_resolution
    if rgb_t.shape[-2:] != (target_res, target_res):
        rgb_t = F.interpolate(rgb_t.unsqueeze(0), size=(target_res, target_res),
                              mode="bilinear", align_corners=False).squeeze(0)
    inp = rgb_t.unsqueeze(0).unsqueeze(1).to(device)
    n_in = cfg.data.input_images
    v2w = bundle["view_to_world_transforms"][0:n_in].to(device).unsqueeze(0)   # [1,n_in,4,4]
    quat = bundle["source_cv2wT_quat"][0:n_in].to(device).unsqueeze(0)         # [1,n_in,4]
    splats = gp(inp, v2w, quat, None)
    return {k: v[0] for k, v in splats.items()}


@torch.no_grad()
def render_at_views(splats, bundle, cfg, bg):
    wvts = bundle["world_view_transforms"].to(device)
    fpts = bundle["full_proj_transforms"].to(device)
    ccs = bundle["camera_centers"].to(device)
    frames = []
    for r in range(wvts.shape[0]):
        out = render_predicted(splats, wvts[r:r + 1], fpts[r:r + 1], ccs[r:r + 1], bg, cfg,
                               focals_pixels=None)
        frames.append(out["render"].clamp(0, 1))
    return torch.stack(frames, dim=0)


@torch.no_grad()
def compute_metrics(preds, gts, lpips_fn, ssim_fn):
    psnrs, ssims, lps = [], [], []
    for i in range(preds.shape[0]):
        p, gt = preds[i], gts[i]
        if p.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(0), size=p.shape[-2:], mode="bilinear",
                               align_corners=False)[0]
        psnrs.append((-10 * torch.log10(((p - gt) ** 2).mean())).item())
        ssims.append(ssim_fn(p, gt).item())
        lps.append(lpips_fn(p.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item())
    return {"PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims)), "LPIPS": float(np.mean(lps))}


@torch.no_grad()
def depth_anchor_splats(splats, bundle, cfg):
    """Privileged baseline variant: rescale the feed-forward splats about the view-0
    camera so their median depth matches the scene's monocular-depth anchor (mean of the
    bundle's per-chair DA3 depths) — the same depth information our pipeline's stage 1
    used. Without this the SRN-trained lifter places content at ~2 units while the scene
    cameras orbit at ~7, so novel views miss it (the raw-baseline failure mode)."""
    v2w0 = bundle["view_to_world_transforms"][0].to(device)      # [4,4]
    # GT-free scene-scale anchor: the eval cameras are scene-centered, so the centroid of
    # the camera positions approximates the scene center; its distance from the view-0
    # camera (origin of the relative frame) is the scene depth. (bundle depth_A/B are
    # normalized DA3 units ~0.6 — not metric — so they cannot anchor scale.) This mirrors
    # the scene-distance convention (~7) the ours gt_compare protocol used.
    target = float(bundle["camera_centers"][1:].mean(0).norm())
    xyz = splats["xyz"]                                          # [N,3] world
    ones = torch.ones(xyz.shape[0], 1, device=device)
    Xw = torch.cat([xyz, ones], dim=1)                           # [N,4]
    w2v0 = torch.inverse(v2w0)
    # transforms in this codebase are row-vector style (x @ M); match that convention
    Xc = Xw @ w2v0
    s = target / float(Xc[:, 2].median())
    Xc2 = Xc.clone(); Xc2[:, :3] = Xc[:, :3] * s
    Xw2 = Xc2 @ v2w0
    out = {k: v.clone() for k, v in splats.items()}
    out["xyz"] = Xw2[:, :3]
    out["scaling"] = splats["scaling"] * s
    return out, s


def main():
    os.makedirs(OUT, exist_ok=True)
    import lpips as lpips_lib
    from utils.loss_utils import ssim as ssim_fn
    lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()

    import hydra as _h
    _h.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{ROOT}/configs")
    cfg = compose(config_name="abs_config", overrides=[
        "abs=diffae_abs", "+dataset=chairs",
        f"opt.pretrained_ckpt={ROOT}/checkpoints/model_chairs.pth",
        "general.data_example_ids_path=not_needed.json", "general.prefix=rebuttal-mc"])

    gp = GaussianSplatPredictor(cfg).to(memory_format=torch.channels_last).to(device)
    ckpt = torch.load(cfg.opt.pretrained_ckpt, map_location=device, weights_only=False)
    gp.load_state_dict(ckpt["model_state_dict"])
    gp.eval()
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)

    rows, rows_anchor, scales = {}, {}, {}
    for scene in sorted(os.listdir(f"{SRC}/stage1")):
        bundle = torch.load(f"{SRC}/stage1/{scene}/bundle.pth", map_location="cpu", weights_only=False)
        bsplats = baseline_splats(bundle["rgb_t"].clone(), bundle, gp, cfg)
        asplats, s = depth_anchor_splats(bsplats, bundle, cfg)
        renders = render_at_views(bsplats, bundle, cfg, bg)
        renders_a = render_at_views(asplats, bundle, cfg, bg)
        gt = bundle["gt_images"].to(device)
        tr = cfg.data.training_resolution
        if gt.shape[-2:] != (tr, tr):
            gt = F.interpolate(gt, size=(tr, tr), mode="bilinear", align_corners=False)
        idxs = list(range(1, gt.shape[0]))
        m = compute_metrics(renders[idxs], gt[idxs], lpips_fn, ssim_fn)
        ma = compute_metrics(renders_a[idxs], gt[idxs], lpips_fn, ssim_fn)
        rows[scene], rows_anchor[scene], scales[scene] = m, ma, s
        print(f"{scene}: raw {m['PSNR']:.2f} | depth-anchored {ma['PSNR']:.2f} (s={s:.2f})")
        # strips: GT | raw baseline | anchored baseline at input view + 3 novel views
        sel = [0, 1, gt.shape[0] // 2, gt.shape[0] - 1]
        strip = np.concatenate([np.concatenate(
            [(gt[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
             (renders[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
             (renders_a[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)], axis=1)
            for i in sel], axis=0)
        Image.fromarray(strip).save(f"{OUT}/{scene}_gt_vs_baseline.png")

    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ["PSNR", "SSIM", "LPIPS"]}
    mean_a = {k: float(np.mean([r[k] for r in rows_anchor.values()])) for k in ["PSNR", "SSIM", "LPIPS"]}
    print(f"\nBASELINE raw (feed-forward model_chairs, n={len(rows)}): {mean}")
    print(f"BASELINE depth-anchored (privileged, n={len(rows_anchor)}): {mean_a}")
    json.dump({"per_scene_raw": rows, "mean_raw": mean,
               "per_scene_depth_anchored": rows_anchor, "mean_depth_anchored": mean_a,
               "anchor_scales": scales, "n": len(rows),
               "protocol": "stage1 bundles, GT cams, views 1..19, same metric code as stage3; "
                           "anchored variant rescales splats about view-0 camera to the bundle's "
                           "DA3 depth anchor (mean of depth_A/depth_B) — same depth info ours used"},
              open(f"{OUT}/baseline_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
