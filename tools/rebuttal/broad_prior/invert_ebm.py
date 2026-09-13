"""TRUE analysis-by-synthesis inversion of the EqM energy-based model.

Given an INPUT OBSERVATION I, recover 3D by guided equilibrium optimization of the EqM
latent z:
    z <- z + eta * ( g_EqM(z)  -  lambda * d/dz || M o (R(Phi(dec(z))) - I) || )
 - g_EqM(z): EqM's equilibrium gradient (energy-descent field) -> PRIOR (kept on the ImageNet
   energy manifold). Used DIRECTLY (no score distillation) -- the reason an EBM is convenient.
 - data term: backprops through renderer -> frozen lifter Phi -> VAE decoder.
Multi-start (K particles), keep best by data loss. Encoder-free warm start = VAE-encode of I
plus random particles. Output: input | inverted reconstruction dec(z*) | recovered 3D turntable.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys, json
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/abs"


def load_obs(spec, cfg):
    """Return I_128 [1,3,128,128] white-bg, M [1,1,128,128] fg mask, tag, gt(optional)."""
    kind, arg = spec.split(":", 1)
    if kind == "png":                          # a pre-masked, centered 128 white-bg image
        im = np.asarray(Image.open(arg).convert("RGB").resize((128, 128))).astype(np.float32) / 255.
        I = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(P.DEV)
        M = (I < 0.985).any(1, keepdim=True).float()
        return I, M, os.path.splitext(os.path.basename(arg))[0], None
    elif kind == "objaverse":
        from splatter_image_datasets.objaverse import ObjaverseDataset
        ds = ObjaverseDataset(cfg, "test"); item = ds[int(arg)]
        I = item["gt_images"][0].unsqueeze(0).to(P.DEV)
        M = (I < 0.985).any(1, keepdim=True).float()
        gt = {"imgs": item["gt_images"].to(P.DEV), "wv": item["world_view_transforms"].to(P.DEV),
              "fp": item["full_proj_transforms"].to(P.DEV), "cc": item["camera_centers"].to(P.DEV)}
        return I, M, f"obj{arg}_{item.get('object_id','')[:8]}", gt
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="png:PATH or objaverse:IDX")
    ap.add_argument("--K", type=int, default=4, help="multi-start particles")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eta", type=float, default=0.0017, help="EqM prior step")
    ap.add_argument("--lam", type=float, default=300.0, help="data weight")
    ap.add_argument("--mu", type=float, default=0.3, help="NAG momentum on prior")
    ap.add_argument("--warm", type=int, default=1, help="include VAE-encode warm start particle")
    ap.add_argument("--tt", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg)
    eqm = P.load_eqm()
    vae = P.load_vae()
    cams = P.build_turntable(cfg, num=args.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()
    I, M, tag, gt = load_obs(args.obs, cfg)
    print(f"observation {tag}")

    K = args.K
    z = torch.randn(K, 4, 32, 32, device=P.DEV)
    if args.warm:
        with torch.no_grad():
            z[0] = P.vae_encode(vae, F.interpolate(I, (256, 256), mode="bilinear", align_corners=False))[0]
    y = torch.full((K,), 1000, device=P.DEV)         # null class -> unconditional energy
    tt0 = torch.zeros(K, device=P.DEV)
    m_nag = torch.zeros_like(z)
    Iexp = I.expand(K, -1, -1, -1)
    Mexp = M.expand(K, -1, -1, -1)

    def data_loss(zc):
        img = P.vae_decode(vae, zc)                  # [K,3,256,256]
        img128 = F.interpolate(img, (128, 128), mode="bilinear", align_corners=False)
        u = img128 * Mexp + (1 - Mexp)               # composite fg on white for the lifter
        rs = []
        for k in range(K):
            sp = P.lift(gp, u[k:k+1])
            rs.append(P.render_view(cfg, sp, cams[0][0], cams[1][0], cams[2][0]))
        r = torch.stack(rs, 0)
        per = ((Mexp * (r - Iexp)) ** 2).flatten(1).mean(1) + \
              0.3 * lp(r * 2 - 1, Iexp * 2 - 1).flatten(1).mean(1)
        return per, r, u

    for t in range(args.steps):
        zc = z.detach().requires_grad_(True)
        per, _, _ = data_loss(zc)
        gL = torch.autograd.grad(per.sum(), zc)[0]
        with torch.no_grad():
            g = P.eqm_grad(eqm, z, y)                # energy-descent field (prior)
            gLn = gL / (gL.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)
            m_nag = args.mu * m_nag + (g - args.lam * args.eta * gLn)
            z = z + args.eta * m_nag
        if t % 50 == 0 or t == args.steps - 1:
            print(f"  step {t}: data loss per-particle {[round(float(x),4) for x in per]}")

    with torch.no_grad():
        per, r, u = data_loss(z.detach())
        best = int(per.argmin())
        zb = z[best:best+1]
        img = P.vae_decode(vae, zb)
        recon = F.interpolate(img, (128, 128), mode="bilinear", align_corners=False)[0]
        ub = u[best]
        sp = P.lift(gp, ub.unsqueeze(0))
    print(f"[{tag}] best particle {best} data loss {float(per[best]):.4f}")
    if gt is not None:
        ps = []
        for v in range(1, gt["wv"].shape[0]):
            with torch.no_grad():
                rr = P.render_view(cfg, sp, gt["wv"][v], gt["fp"][v], gt["cc"][v])
            ps.append(float(-10 * torch.log10(((rr - gt["imgs"][v]) ** 2).mean() + 1e-12)))
        print(f"[{tag}] inverted novel-view PSNR {np.mean(ps):.2f}")

    tt = P.turntable_frames(cfg, sp, cams)
    inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rec = (recon.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, rec, f], axis=1) for f in tt]
    imageio.mimsave(f"{OUT}/abs_{tag}.mp4", frames, fps=20, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/abs_{tag}_strip.png")
    print(f"saved abs_{tag}.mp4  (INPUT | inverted reconstruction dec(z*) | recovered 3D)")


if __name__ == "__main__":
    main()
