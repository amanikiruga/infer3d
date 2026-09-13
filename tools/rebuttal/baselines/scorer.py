"""Shared Task-A scorer + gaussian-render utility.

score_views(): PSNR/SSIM/LPIPS between rendered PNGs and target PNGs — formulas
copied from create_loop_and_eval (the code that produced the paper numbers).
render_relative(): renders splats (already in the INPUT-camera frame) at the
exported relative cameras — used by ours/Splatter and any gaussians-producing
baseline, and by the validation check.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os
import sys
from math import exp

import numpy as np
import torch
import torch.nn.functional as F

WORKTREE = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, WORKTREE)


# Self-contained SSIM (verbatim from splatter utils/loss_utils.py) so scoring
# works regardless of which baseline repo's `utils` package shadows sys.path.
def _gaussian(ws, sigma):
    g = torch.Tensor([exp(-(x - ws // 2) ** 2 / float(2 * sigma ** 2)) for x in range(ws)])
    return g / g.sum()


def _create_window(ws, ch):
    _1d = _gaussian(ws, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(ch, 1, ws, ws).contiguous()


def ssim_metric(img1, img2, ws=11):
    ch = img1.size(-3)
    window = _create_window(ws, ch).to(img1.device).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=ws // 2, groups=ch)
    mu2 = F.conv2d(img2, window, padding=ws // 2, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=ws // 2, groups=ch) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=ws // 2, groups=ch) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=ws // 2, groups=ch) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.mean()


_lpips = None


def _lpips_fn():
    global _lpips
    if _lpips is None:
        import lpips as lpips_lib
        _lpips = lpips_lib.LPIPS(net="vgg").cuda()
    return _lpips


def load_png(p):  # -> [3,H,W] float 0..1 torch
    from PIL import Image
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def score_views(render_paths, target_paths, device="cuda"):
    """Per-object metrics, averaged over views (identical to create_loop_and_eval)."""
    ssim_fn = ssim_metric
    lp = _lpips_fn()
    psnr_all, ssim_all, lpips_all = [], [], []
    for rp, tp in zip(render_paths, target_paths):
        img = load_png(rp).to(device)
        gt = load_png(tp).to(device)
        with torch.no_grad():
            lpips_v = lp(img.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((img - gt) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(img, gt).item()
        psnr_all.append(psnr); ssim_all.append(ssim); lpips_all.append(lpips_v)
    return {"PSNR": float(np.mean(psnr_all)), "SSIM": float(np.mean(ssim_all)),
            "LPIPS": float(np.mean(lpips_all)), "n_views": len(psnr_all)}


def render_relative(splats, bundle_dir, out_dir, cfg, device="cuda"):
    """Render splats (input-camera frame) at the bundle's relative cams -> PNGs."""
    from PIL import Image
    from gaussian_renderer import render_predicted
    cams = np.load(f"{bundle_dir}/cams_relative.npz")
    os.makedirs(out_dir, exist_ok=True)
    bg = torch.tensor([1., 1., 1.], device=device)
    outs = []
    for i in range(cams["world_view_transforms"].shape[0]):
        wvt = torch.from_numpy(cams["world_view_transforms"][i]).float().to(device)
        fpt = torch.from_numpy(cams["full_proj_transforms"][i]).float().to(device)
        cc = torch.from_numpy(cams["camera_centers"][i]).float().to(device)
        with torch.no_grad():
            img = render_predicted(splats, wvt, fpt, cc, bg, cfg,
                                   focals_pixels=None)["render"]
        arr = np.clip(img.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        p = f"{out_dir}/{i:02d}.png"
        Image.fromarray(arr).save(p)
        outs.append(p)
    return outs


def bundle_targets(bundle_dir):
    t = sorted(os.listdir(f"{bundle_dir}/targets"))
    return [f"{bundle_dir}/targets/{x}" for x in t]
