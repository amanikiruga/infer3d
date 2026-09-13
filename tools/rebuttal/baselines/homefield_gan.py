"""Homefield NVS for the GAN-inversion baselines (EG3D+PTI, FINV-SV) — each method
evaluated in its OWN generator domain, mirroring the comprehensive treatment NFI got.

Why this exists: NFI is the only baseline whose pose frame is dataset-aligned, so it
was the only one with a calibrated in-distribution NVS number. Here we remove the
calibration obstacle entirely: sample K latents from the method's own pretrained
generator, render 1 input view + 24 target views at cameras WE choose (poses known
exactly in the generator frame — nothing to calibrate). Hand the method only the
input image, run its full inversion pipeline, render at the true target cameras,
PSNR vs the generator's own target renders. The target is ON the prior manifold —
the most favorable setting that exists for these methods.

  own-pose    method estimates the input pose itself; targets anchored relative to
              its estimate (same protocol as task A / NFI own-pose)
  oracle-pose pose frozen at the true input camera, latent-only + PTI (mirrors
              nfi_oracle's invert_fixed)

  python homefield_gan.py --gen                      # make the K sample bundles
  python homefield_gan.py --method eg3d [--oracle]
  python homefield_gan.py --method finv
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
import finv_sv as FV
dev = "cuda"
HF = f"{WT}/rebuttal/results/homefield"
INP = f"{HF}/inputs_eg3d"
K = 15
N_TGT = 24
EL = np.pi / 2 - 0.3  # slightly above the car's horizon

# This checkpoint's canonical frame has cars z-up, but LookAtPoseSampler builds y-up
# cameras (its equator ring orbits over the roof). Conjugate the whole camera family by
# a fixed Rx(90) so (az, el) parameterizes the z-up ring the generator was trained on.
# Patching E.make_c also reroutes finv_sv (same module object), so the pose SEARCH of
# both methods runs in this family too — the true input pose is inside it (no handicap).
ROT = torch.tensor([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]],
                   device=dev)
_make_c_y = E.make_c


def _make_c_z(az, el, G, radius=None):
    c25, c2w = _make_c_y(az, el, G, radius)
    c2w = ROT[None] @ c2w
    return torch.cat([c2w.reshape(1, 16), c25[:, 16:]], 1), c2w


E.make_c = _make_c_z


def psnr(a, b):  # uint8 arrays
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def render_at(G, ws, c2w):
    c = np.concatenate([np.asarray(c2w, np.float32).reshape(16),
                        E.INTR.numpy().reshape(9)])[None].astype(np.float32)
    with torch.no_grad():
        img = E.synth(G, ws, torch.from_numpy(c).to(dev))[0]
    return np.clip((img.permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)


def cam(G, az, el):
    _, c2w = E.make_c(torch.tensor(float(az), device=dev), torch.tensor(float(el), device=dev), G)
    return c2w[0].detach().cpu().numpy()


def gen_samples():
    from PIL import Image
    G = E.load_G()
    for k in range(K):
        oid = f"s{k:02d}"; od = f"{INP}/{oid}"; os.makedirs(f"{od}/targets", exist_ok=True)
        torch.manual_seed(1000 + k)
        z = torch.randn(1, G.z_dim, device=dev)
        with torch.no_grad():
            ws = G.mapping(z, torch.zeros(1, G.c_dim, device=dev), truncation_psi=0.7)
        az_in = 2 * np.pi * k / K + 0.3
        az_tgt = [2 * np.pi * i / N_TGT for i in range(N_TGT)]
        Image.fromarray(render_at(G, ws, cam(G, az_in, EL))).save(f"{od}/input.png")
        for i, a in enumerate(az_tgt):
            Image.fromarray(render_at(G, ws, cam(G, a, EL))).save(f"{od}/targets/{i:02d}.png")
        np.save(f"{od}/w_true.npy", ws.cpu().numpy())  # debugging only; inversion never reads it
        json.dump({"az_in": az_in, "el_in": EL, "az_tgt": az_tgt, "el_tgt": EL},
                  open(f"{od}/cams.json", "w"))
        print(f"gen {oid}")
    print("GEN DONE")


def invert_oracle(G, lp, target, az_t, el_t, main_steps=350, pti_steps=200):
    """Latent-only inversion + PTI with pose frozen at the TRUE input camera."""
    wa = E.w_avg(G)
    c_fixed, c2w = E.make_c(torch.tensor(float(az_t), device=dev),
                            torch.tensor(float(el_t), device=dev), G)
    c_fixed = c_fixed.detach()
    ws = wa.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([ws], lr=8e-3)
    for _ in range(main_steps):
        img = E.synth(G, ws, c_fixed)
        loss = lp(img, target).mean() + 0.1 * F.mse_loss(img, target)
        opt.zero_grad(); loss.backward(); opt.step()
    ws = ws.detach()
    for p in G.parameters():
        p.requires_grad_(True)
    optG = torch.optim.Adam(G.parameters(), lr=3e-4)
    for _ in range(pti_steps):
        out = G.synthesis(ws, c_fixed, noise_mode="const", force_fp32=True)
        img, img_raw = out["image"], out["image_raw"]
        tgt_raw = F.interpolate(target, size=img_raw.shape[-1], mode="area")
        loss = (lp(img, target).mean() + F.mse_loss(img, target)
                + lp(img_raw, tgt_raw).mean() + F.mse_loss(img_raw, tgt_raw))
        optG.zero_grad(); loss.backward(); optG.step()
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return ws


def run(method, oracle, only=None):
    from PIL import Image
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(dev)
    tag = f"{method}_oracle" if oracle else method
    ids = [only] if only else sorted(os.listdir(INP))
    all_psnr = {}
    for oid in ids:
        od = f"{HF}/{tag}/{oid}"; os.makedirs(f"{od}/renders", exist_ok=True)
        if os.path.exists(f"{od}/meta.json"):
            all_psnr[oid] = json.load(open(f"{od}/meta.json"))["psnr"]
            print(f"skip {oid} (done, {all_psnr[oid]:.2f})"); continue
        cams = json.load(open(f"{INP}/{oid}/cams.json"))
        target = E.load_input(f"{INP}/{oid}/input.png")
        G = E.load_G()  # fresh per object: PTI mutates weights
        c_in_true = cam(G, cams["az_in"], cams["el_in"])
        if oracle:
            ws = invert_oracle(G, lp, target, cams["az_in"], cams["el_in"])
            c2w_hat = c_in_true
        elif method == "eg3d":
            ws, c2w_hat, _ = E.invert(G, lp, target)
        else:  # finv
            ws, c2w_hat, info = FV.finv_invert(G, lp, target)
            json.dump(info, open(f"{od}/finv_info.json", "w"))
        ps = []
        for i, a in enumerate(cams["az_tgt"]):
            c_tgt_true = cam(G, a, cams["el_tgt"])
            ct = c2w_hat @ np.linalg.inv(c_in_true) @ c_tgt_true  # own-pose: relative anchor
            r = render_at(G, ws, ct)
            Image.fromarray(r).save(f"{od}/renders/{i:02d}.png")
            gt = np.asarray(Image.open(f"{INP}/{oid}/targets/{i:02d}.png").convert("RGB"))
            ps.append(psnr(r, gt))
        m = float(np.mean(ps))
        json.dump({"psnr": m, "per_view": [round(p, 2) for p in ps],
                   "c2w_hat": np.asarray(c2w_hat).tolist()}, open(f"{od}/meta.json", "w"))
        all_psnr[oid] = m
        print(f"done {oid} psnr={m:.2f}")
        del G; torch.cuda.empty_cache()
    if not only:
        json.dump({"mean": float(np.mean(list(all_psnr.values()))), "per_object": all_psnr},
                  open(f"{HF}/{tag}_summary.json", "w"), indent=1)
        print(f"[{tag}] mean PSNR = {np.mean(list(all_psnr.values())):.2f} over {len(all_psnr)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--method", choices=["eg3d", "finv"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.gen:
        gen_samples()
    else:
        run(a.method, a.oracle, a.only)
