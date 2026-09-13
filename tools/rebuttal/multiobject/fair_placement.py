"""FAIR multichair placement refinement (no test-view GT).

The pipeline's composition uses a single hand-tuned camera_scale (÷7) and no per-chair
frame correction; the per-chair oracle showed correct per-object placement lifts the
ACTUAL reconstructions to ~18.4 mean, i.e. the 14->18 gap is placement, not blur. Here we
recover it FAIRLY: optimize each chair's rigid placement (7 DoF, init identity) using ONLY
information the method has at test time —
  (1) the INPUT view (view 0 == rgb_t): composed render must reproduce the input image
      (MSE + LPIPS), exactly the single-object analysis-by-synthesis objective;
  (2) a monocular-depth relative anchor: the two chairs' camera-space depths must match the
      ratio of their stage-1 DepthAnything-3 depths (breaks the single-view scale/depth
      degeneracy that view 0 alone cannot resolve).
No novel-view GT is ever touched. We then REPORT novel-view PSNR/SSIM/LPIPS over views 1..19.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os, sys
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

ROOT = f"{_EXT}/splatter-image"
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.append(ROOT)
sys.path.insert(0, f"{ROOT}/experiments/neurips_submission/multichair_pipeline")

import hydra
from hydra import initialize_config_dir, compose
from gaussian_renderer import render_predicted
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import quaternion_raw_multiply, matrix_to_quaternion
import lpips as lpips_lib
from utils.loss_utils import ssim as ssim_fn
import multichair_stage3_gt_compare as G

device = "cuda"
SRC = f"{ROOT}/checkpoints-multichair-pipeline-v1-20scenes"
OUT = f"{WT}/rebuttal/multiobject/fair_placement"


def sixd_to_R(x):
    a1, a2 = x[:3], x[3:]
    b1 = F.normalize(a1, dim=0)
    b2 = F.normalize(a2 - (b1 * a2).sum() * b1, dim=0)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=0)], dim=1)


def transform_splats(sp, s, R, t):
    out = {k: v for k, v in sp.items()}
    out["xyz"] = s * (sp["xyz"] @ R.T) + t
    out["scaling"] = sp["scaling"] * s
    qR = matrix_to_quaternion(R).reshape(1, 4)
    out["rotation"] = quaternion_raw_multiply(qR.expand(sp["rotation"].shape[0], 4), sp["rotation"])
    return out


def build_cams(bundle, camera_scale):
    wvts = bundle["world_view_transforms"].to(device).clone()
    wvts[:, 3, :3] = wvts[:, 3, :3] * camera_scale
    proj = getProjectionMatrix(znear=G.CHAIRS_ZNEAR, zfar=G.CHAIRS_ZFAR,
                               fovX=G.CHAIRS_FOV_DEG * 2 * np.pi / 360,
                               fovY=G.CHAIRS_FOV_DEG * 2 * np.pi / 360).transpose(0, 1).to(device)
    fpts = torch.stack([wvts[i].unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0) for i in range(wvts.shape[0])])
    ccs = torch.stack([wvts[i].inverse()[3, :3] for i in range(wvts.shape[0])])
    return wvts, fpts, ccs


def cam_depth(sp, wvt):
    """opacity-weighted mean camera-space z of the splat centres at camera wvt (row-major)."""
    xyz = sp["xyz"]
    h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=device)], 1)
    z = (h @ wvt)[:, 2]
    w = torch.sigmoid(sp["opacity"][:, 0]) if sp["opacity"].dim() > 1 else torch.sigmoid(sp["opacity"])
    return (z * w).sum() / (w.sum() + 1e-6)


def main():
    os.makedirs(OUT, exist_ok=True)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{ROOT}/configs")
    cfg = compose(config_name="abs_config", overrides=[
        "abs=diffae_abs", "+dataset=chairs",
        f"opt.pretrained_ckpt={ROOT}/checkpoints/model_chairs.pth",
        "general.data_example_ids_path=not_needed.json", "general.prefix=fair-mc"])
    cfgr = deepcopy(cfg)
    cfgr.data.fov = G.CHAIRS_FOV_DEG; cfgr.data.znear = G.CHAIRS_ZNEAR; cfgr.data.zfar = G.CHAIRS_ZFAR
    gp = G.load_splatter_image(cfg)
    bg = torch.tensor([1., 1, 1], device=device)
    lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()

    res = {}
    for scene in sorted(os.listdir(f"{SRC}/stage1")):
        bundle = torch.load(f"{SRC}/stage1/{scene}/bundle.pth", map_location="cpu", weights_only=False)
        pa = torch.load(f"{SRC}/stage2/{scene}/chair0/final.pth", map_location="cpu", weights_only=False)
        pb = torch.load(f"{SRC}/stage2/{scene}/chair1/final.pth", map_location="cpu", weights_only=False)
        sa = {k: v.detach() for k, v in G.reconstruct_chair_splats(pa, gp, cfg).items()}
        sb = {k: v.detach() for k, v in G.reconstruct_chair_splats(pb, gp, cfg).items()}

        gt = bundle["gt_images"].to(device)
        if gt.shape[-2:] != (128, 128):
            gt = F.interpolate(gt, size=(128, 128), mode="bilinear", align_corners=False)
        cz = 0.5 * (float(sa["xyz"][:, 2].mean()) + float(sb["xyz"][:, 2].mean()))
        wvts, fpts, ccs = build_cams(bundle, cz / 7.0)
        inp = gt[0]                                        # view 0 == input image
        mA = bundle["mask_A"].reshape(1, 128, 128).to(device)
        mB = bundle["mask_B"].reshape(1, 128, 128).to(device)
        depth_ratio_gt = float(bundle["depth_A"]) / float(bundle["depth_B"])
        d0a, d0b = float(cam_depth(sa, wvts[0])), float(cam_depth(sb, wvts[0]))
        dmean0 = 0.5 * (d0a + d0b)                          # trust global scale from ÷7

        # TRANSLATION-ONLY per chair (well-posed: masks fix x,y+apparent size, DA3 ratio
        # fixes relative depth; scale/rotation frozen so a single view cannot overfit).
        params = {n: {"t": torch.zeros(3, device=device, requires_grad=True)} for n in "ab"}
        opt = torch.optim.Adam([params["a"]["t"], params["b"]["t"]], lr=0.01)
        I3 = torch.eye(3, device=device)

        def tf(name, sp):
            return transform_splats(sp, torch.ones((), device=device), I3, params[name]["t"])

        def render(sp, i):
            return render_predicted(sp, wvts[i:i+1], fpts[i:i+1], ccs[i:i+1], bg, cfgr,
                                    focals_pixels=None)["render"].clamp(0, 1)

        for it in range(400):
            opt.zero_grad()
            ta, tb = tf("a", sa), tf("b", sb)
            rA, rB = render(ta, 0), render(tb, 0)          # each chair alone at input view
            # per-chair MASKED photometric: chair stays in its own input region
            lA = (((rA - inp) ** 2) * mA).sum() / (mA.sum() + 1e-6)
            lB = (((rB - inp) ** 2) * mB).sum() / (mB.sum() + 1e-6)
            da, db = cam_depth(ta, wvts[0]), cam_depth(tb, wvts[0])
            l_ratio = (torch.log(da / db.clamp(min=1e-3)) - np.log(depth_ratio_gt)) ** 2
            l_mean = (0.5 * (da + db) - dmean0) ** 2       # keep global scale (÷7 trusted)
            loss = lA + lB + 0.3 * l_ratio + 0.3 * l_mean
            loss.backward()
            opt.step()

        # report novel views 1..19 (no GT used in optimization)
        with torch.no_grad():
            merged = G.merge(tf("a", sa), tf("b", sb))
            idx = list(range(1, gt.shape[0]))
            preds = torch.stack([render(merged, i) for i in idx])
            g = gt[idx]
            psnr = float(-10 * torch.log10(((preds - g) ** 2).mean() + 1e-12))
            per = [float(-10 * torch.log10(((preds[k] - g[k]) ** 2).mean() + 1e-12)) for k in range(len(idx))]
            ss = float(np.mean([ssim_fn(preds[k], g[k]).item() for k in range(len(idx))]))
            lp = float(np.mean([lpips_fn(preds[k].unsqueeze(0) * 2 - 1, g[k].unsqueeze(0) * 2 - 1).item()
                                for k in range(len(idx))]))
            r0 = render(merged, 0)
            in_psnr = float(-10 * torch.log10(((r0 - inp) ** 2).mean() + 1e-12))
        res[scene] = {"psnr": psnr, "ssim": ss, "lpips": lp, "input_psnr": in_psnr}
        print(f"{scene[:20]}: novel PSNR {psnr:.2f}  SSIM {ss:.3f}  LPIPS {lp:.3f}  (input-fit {in_psnr:.1f})")

    P = np.array([v["psnr"] for v in res.values()])
    summary = {"n": len(P), "mean_psnr": float(P.mean()), "median_psnr": float(np.median(P)),
               "mean_ssim": float(np.mean([v["ssim"] for v in res.values()])),
               "mean_lpips": float(np.mean([v["lpips"] for v in res.values()])),
               "ge18": int((P >= 18).sum()), "ge16": int((P >= 16).sum()), "per_scene": res}
    print(json.dumps({k: summary[k] for k in ["n", "mean_psnr", "median_psnr", "mean_ssim", "mean_lpips", "ge18", "ge16"]}, indent=1))
    json.dump(summary, open(f"{OUT}/fair_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
