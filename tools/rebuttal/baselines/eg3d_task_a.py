"""3D-GAN-Inversion / EG3D-PTI (Ko et al. WACV'23) on Task A (NMR cars, SO(3)).

Single-image inversion of the pretrained EG3D ShapeNet-cars generator with the
method's two ingredients the AC cites: (1) camera-POSE OPTIMIZATION (no GT pose --
we optimize azimuth/elevation from multi-azimuth restarts, matching the paper's
estimate+optimize idea) and (2) PTI generator fine-tuning. Reconstructs in EG3D's
canonical frame with an estimated input camera; renders our GT target cameras
(anchored at the estimated input pose, convention calibrated on the control).

  python eg3d_task_a.py [--only OBJ] [--bundles ...] [--out ...] [--flip s,s,s]
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = f"{_EXT}/rebuttal_infer3d_baselines/3D-GAN-Inversion"
CKPT = f"{_EXT}/rebuttal_infer3d_baselines/checkpoints/shapenetcars128-64.pkl"
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, REPO)

from torch_utils.ops import bias_act, upfirdn2d
bias_act._init = lambda: False
upfirdn2d._init = lambda: False
import pickle
from utils.camera_utils import LookAtPoseSampler
import lpips as lpips_lib

device = "cuda"
RES = 128
INTR = torch.tensor([[1.0254, 0, 0.5], [0, 1.0254, 0.5], [0, 0, 1.]])  # SRN-cars normalized


def load_G():
    with open(CKPT, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device).eval().float()
    # EG3D's SR modifies rgb_image (a view of the 32-ch feature) in place -> breaks
    # autograd through pose/w. Clone SR inputs to make backprop valid (no numeric change).
    sr = G.superresolution
    _orig = sr.forward
    sr.forward = lambda rgb, x, ws, **kw: _orig(rgb.clone(), x.clone(), ws, **kw)
    return G


def make_c(az, el, G, radius=None):
    """cam2world+intrinsics 25-vec from azimuth/elevation (radians). Differentiable."""
    radius = radius or G.rendering_kwargs.get("avg_camera_radius", 1.7)
    piv = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
    c2w = LookAtPoseSampler.sample(az, el, piv, radius=radius, device=device)  # (1,4,4)
    intr = INTR.to(device).reshape(1, 9)
    return torch.cat([c2w.reshape(1, 16), intr], 1), c2w


def load_input(png):
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.float32) / 255. * 2 - 1
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)  # (1,3,H,W) [-1,1]


def w_avg(G, n=5000, psi=0.7):
    z = torch.randn(n, G.z_dim, device=device)
    c = torch.zeros(n, G.c_dim, device=device)  # c ignored (c_gen_conditioning_zero)
    with torch.no_grad():
        w = G.mapping(z, c, truncation_psi=1.0)
    return w.mean(0, keepdim=True)  # (1, num_ws, 512)


def synth(G, ws, c):
    return G.synthesis(ws, c, noise_mode="const", force_fp32=True)["image"]


def invert(G, lp, target, restarts=(0., np.pi/2, np.pi, 3*np.pi/2),
           preheat=60, main_steps=350, pti_steps=200):
    """Returns ws (1,num_ws,512), (az,el) estimated, and does PTI on G (in place)."""
    wa = w_avg(G)
    # ---- pose preheat: multi-azimuth, optimize az/el (w frozen at w_avg) ----
    best = None
    for az0 in restarts:
        az = torch.tensor(float(az0), device=device, requires_grad=True)
        el = torch.tensor(float(np.pi/2), device=device, requires_grad=True)
        opt = torch.optim.Adam([az, el], lr=0.05)
        for _ in range(preheat):
            c, _ = make_c(az, el, G)
            img = synth(G, wa, c)
            loss = lp(img, target).mean() + 0.1 * F.mse_loss(img, target)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            c, _ = make_c(az, el, G)
            l = (lp(synth(G, wa, c), target).mean()).item()
        if best is None or l < best[0]:
            best = (l, az.detach().item(), el.detach().item())
    az = torch.tensor(best[1], device=device, requires_grad=True)
    el = torch.tensor(best[2], device=device, requires_grad=True)

    # ---- joint w + pose optimization ----
    ws = wa.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [ws], "lr": 8e-3},
                            {"params": [az, el], "lr": 7e-3}])
    for _ in range(main_steps):
        c, _ = make_c(az, el, G)
        img = synth(G, ws, c)
        loss = lp(img, target).mean() + 0.1 * F.mse_loss(img, target)
        opt.zero_grad(); loss.backward(); opt.step()
    ws = ws.detach()
    az_f, el_f = az.detach(), el.detach()

    # ---- PTI: fine-tune all G params (pose+w frozen) ----
    c_fixed, c2w_hat = make_c(az_f, el_f, G)
    c_fixed = c_fixed.detach()
    for p in G.parameters():
        p.requires_grad_(True)
    optG = torch.optim.Adam(G.parameters(), lr=3e-4)
    for _ in range(pti_steps):
        out = G.synthesis(ws, c_fixed, noise_mode="const", force_fp32=True)
        img = out["image"]
        img_raw = out["image_raw"]
        tgt_raw = F.interpolate(target, size=img_raw.shape[-1], mode="area")
        loss = (lp(img, target).mean() + F.mse_loss(img, target)
                + lp(img_raw, tgt_raw).mean() + F.mse_loss(img_raw, tgt_raw))
        optG.zero_grad(); loss.backward(); optG.step()
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return ws, c2w_hat.detach()[0].cpu().numpy(), best[0]


def save_rgb(img, path):  # img (3,H,W) [-1,1]
    from PIL import Image
    a = np.clip((img.permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def run_object(G0_state, lp, bdir, odir, flip, main_steps, pti_steps):
    # fresh G per object (PTI mutates weights)
    G = load_G()
    cabs = np.load(f"{bdir}/cams_absolute.npz")
    c_in_cv = cabs["input_v2w"].T.astype(np.float32)
    targets_v2w = cabs["targets_v2w"]
    target = load_input(f"{bdir}/input.png")
    ws, c2w_hat, pre_loss = invert(G, lp, target, main_steps=main_steps, pti_steps=pti_steps)
    Fm = np.diag(list(flip) + [1]).astype(np.float32)
    os.makedirs(f"{odir}/renders", exist_ok=True)
    for i in range(targets_v2w.shape[0]):
        c_tgt_cv = targets_v2w[i].T.astype(np.float32)
        ct = c2w_hat @ Fm @ np.linalg.inv(c_in_cv) @ c_tgt_cv @ Fm
        c = np.concatenate([ct.reshape(16), INTR.numpy().reshape(9)]).astype(np.float32)
        c_t = torch.from_numpy(c)[None].to(device)
        with torch.no_grad():
            img = synth(G, ws, c_t)[0]
        save_rgb(torch.clamp(img, -1, 1), f"{odir}/renders/{i:02d}.png")
    # reproj at estimated pose
    c_hat = np.concatenate([c2w_hat.reshape(16), INTR.numpy().reshape(9)]).astype(np.float32)
    with torch.no_grad():
        save_rgb(torch.clamp(synth(G, ws, torch.from_numpy(c_hat)[None].to(device))[0], -1, 1),
                 f"{odir}/reproj.png")
    json.dump({"pre_loss": pre_loss, "flip": list(flip), "c2w_hat": c2w_hat.tolist()},
              open(f"{odir}/meta.json", "w"))
    del G
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_runs/eg3d")
    ap.add_argument("--only", default=None)
    ap.add_argument("--flip", default="1,-1,-1")
    ap.add_argument("--main_steps", type=int, default=350)
    ap.add_argument("--pti_steps", type=int, default=200)
    args = ap.parse_args()
    flip = [int(x) for x in args.flip.split(",")]
    lp = lpips_lib.LPIPS(net="vgg").to(device)
    ids = [args.only] if args.only else sorted(os.listdir(args.bundles))
    for oid in ids:
        odir = f"{args.out}/{oid}"
        os.makedirs(odir, exist_ok=True)
        run_object(None, lp, f"{args.bundles}/{oid}", odir, flip,
                   args.main_steps, args.pti_steps)
        print(f"done {oid}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
