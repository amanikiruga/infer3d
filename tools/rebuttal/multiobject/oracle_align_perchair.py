"""DIAGNOSTIC: how high can multichair PSNR go if the composed scene's global frame
(scale / rotation / translation) were aligned perfectly, holding the actual per-chair
reconstructions fixed? This is an UPPER BOUND (fit to test views) — NOT a reportable
number — used only to decide whether the 14->18 gap is frame/placement (fixable, fair)
or reconstruction-bound (blur/pose, prior-limited).

Rebuilds each scene's merged splats exactly as multichair_stage3_gt_compare.py, then
optimizes a global similarity (log-scale s, 6D rotation R, translation t) applied to the
Gaussian centers + scales + orientation quaternions, rendered at the GT scene cameras
(chairs-FOV path, identical to gt_compare), against novel views 1..19 (PSNR loss).
Init at the pipeline's own camera_scale heuristic so the optimizer can only improve.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os, sys
from pathlib import Path
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
import multichair_stage3_gt_compare as G

device = "cuda"
SRC = f"{ROOT}/checkpoints-multichair-pipeline-v1-20scenes"
OUT = f"{WT}/rebuttal/multiobject/oracle_align_perchair"


def sixd_to_R(x):  # Zhou et al. 6D -> rotation
    a1, a2 = x[:3], x[3:]
    b1 = F.normalize(a1, dim=0)
    b2 = F.normalize(a2 - (b1 * a2).sum() * b1, dim=0)
    b3 = torch.cross(b1, b2, dim=0)
    return torch.stack([b1, b2, b3], dim=1)  # columns


def transform_splats(sp, s, R, t):
    out = {k: v for k, v in sp.items()}
    xyz = sp["xyz"]                                  # [N,3]
    out["xyz"] = s * (xyz @ R.T) + t
    out["scaling"] = sp["scaling"] * s
    qR = matrix_to_quaternion(R).reshape(1, 4)       # [1,4]
    out["rotation"] = quaternion_raw_multiply(qR.expand(sp["rotation"].shape[0], 4), sp["rotation"])
    return out


def build_cams(bundle, camera_scale):
    """Exactly gt_compare.render_at_scene_views: scale GT camera positions by
    camera_scale, chairs-FOV projection. Init here == the pipeline's 14.04 config."""
    wvts = bundle["world_view_transforms"].to(device).clone()
    wvts[:, 3, :3] = wvts[:, 3, :3] * camera_scale
    n = wvts.shape[0]
    proj = getProjectionMatrix(znear=G.CHAIRS_ZNEAR, zfar=G.CHAIRS_ZFAR,
                               fovX=G.CHAIRS_FOV_DEG * 2 * np.pi / 360,
                               fovY=G.CHAIRS_FOV_DEG * 2 * np.pi / 360).transpose(0, 1).to(device)
    fpts, ccs = [], []
    for i in range(n):
        fpts.append(wvts[i].unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0))
        ccs.append(wvts[i].inverse()[3, :3])
    return wvts, torch.stack(fpts), torch.stack(ccs)


def main():
    os.makedirs(OUT, exist_ok=True)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{ROOT}/configs")
    cfg = compose(config_name="abs_config", overrides=[
        "abs=diffae_abs", "+dataset=chairs",
        f"opt.pretrained_ckpt={ROOT}/checkpoints/model_chairs.pth",
        "general.data_example_ids_path=not_needed.json", "general.prefix=oracle-mc"])
    from copy import deepcopy
    cfgr = deepcopy(cfg)
    cfgr.data.fov = G.CHAIRS_FOV_DEG; cfgr.data.znear = G.CHAIRS_ZNEAR; cfgr.data.zfar = G.CHAIRS_ZFAR
    gp = G.load_splatter_image(cfg)
    bg = torch.tensor([1., 1, 1], device=device)

    res = {}
    for scene in sorted(os.listdir(f"{SRC}/stage1")):
        bundle = torch.load(f"{SRC}/stage1/{scene}/bundle.pth", map_location="cpu", weights_only=False)
        pa = torch.load(f"{SRC}/stage2/{scene}/chair0/final.pth", map_location="cpu", weights_only=False)
        pb = torch.load(f"{SRC}/stage2/{scene}/chair1/final.pth", map_location="cpu", weights_only=False)
        sa = G.reconstruct_chair_splats(pa, gp, cfg)
        sb = G.reconstruct_chair_splats(pb, gp, cfg)
        sa = {k: v.detach() for k, v in sa.items()}
        sb = {k: v.detach() for k, v in sb.items()}
        merged = G.merge(sa, sb)

        gt = bundle["gt_images"].to(device)
        if gt.shape[-2:] != (128, 128):
            gt = F.interpolate(gt, size=(128, 128), mode="bilinear", align_corners=False)
        cz = 0.5 * (float(sa["xyz"][:, 2].mean().detach()) + float(sb["xyz"][:, 2].mean().detach()))
        camera_scale = cz / 7.0
        wvts, fpts, ccs = build_cams(bundle, camera_scale)
        eval_idx = list(range(1, gt.shape[0]))
        g = gt[eval_idx]

        # INDEPENDENT residual similarity per chair (2x7=14 DoF), each init IDENTITY so
        # init == the pipeline's own 14.04 render. Upper bound if each object were placed
        # perfectly (but with its ACTUAL reconstructed shape/appearance).
        params = {}
        for name in ("a", "b"):
            params[name] = {
                "log_s": torch.zeros((), device=device, requires_grad=True),
                "six": torch.tensor([1., 0, 0, 0, 1, 0], device=device, requires_grad=True),
                "t": torch.zeros(3, device=device, requires_grad=True)}
        flat = [p for d in params.values() for p in d.values()]
        opt = torch.optim.Adam([{"params": [params["a"]["six"], params["b"]["six"]], "lr": 0.01},
                                {"params": [params["a"]["log_s"], params["a"]["t"],
                                            params["b"]["log_s"], params["b"]["t"]], "lr": 0.005}])

        def compose_scene():
            ta = transform_splats(sa, torch.exp(params["a"]["log_s"]), sixd_to_R(params["a"]["six"]), params["a"]["t"])
            tb = transform_splats(sb, torch.exp(params["b"]["log_s"]), sixd_to_R(params["b"]["six"]), params["b"]["t"])
            return G.merge(ta, tb)

        def render_eval(sp):
            outs = [render_predicted(sp, wvts[i:i+1], fpts[i:i+1], ccs[i:i+1], bg, cfgr,
                                     focals_pixels=None)["render"].clamp(0, 1) for i in eval_idx]
            return torch.stack(outs)

        with torch.no_grad():
            init_psnr = float(-10 * torch.log10(((render_eval(merged) - g) ** 2).mean() + 1e-12))
        best = init_psnr
        for it in range(400):
            opt.zero_grad()
            pred = render_eval(compose_scene())
            mse = ((pred - g) ** 2).mean()
            mse.backward()
            opt.step()
            with torch.no_grad():
                best = max(best, float(-10 * torch.log10(((pred - g) ** 2).mean() + 1e-12)))
        res[scene] = {"camera_scale": camera_scale, "init_psnr": init_psnr, "oracle_psnr": best}
        print(f"{scene[:20]}: init {init_psnr:.2f} -> perchair-oracle {best:.2f}  (+{best-init_psnr:.2f})")

    P = np.array([v["oracle_psnr"] for v in res.values()])
    I = np.array([v["init_psnr"] for v in res.values()])
    summary = {"n": len(P), "init_mean": float(I.mean()), "oracle_mean": float(P.mean()),
               "oracle_median": float(np.median(P)), "gain_mean": float((P - I).mean()),
               "ge18": int((P >= 18).sum()), "ge16": int((P >= 16).sum()), "per_scene": res}
    print(json.dumps({k: summary[k] for k in ["n", "init_mean", "oracle_mean", "oracle_median", "gain_mean", "ge18", "ge16"]}, indent=1))
    json.dump(summary, open(f"{OUT}/oracle_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
