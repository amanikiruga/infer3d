"""PARTIAL-OBSERVATION Infer3D inversion -- real posterior sampling with an EqM prior.

The observation is PARTIAL: a region of the object is occluded (unobserved). The loss is
supervised ONLY on the visible region; the EqM energy prior must SAMPLE the hidden content
so that (a) the visible part matches and (b) the whole is a plausible on-manifold object that
lifts to coherent 3D. This is genuine analysis-by-synthesis (not equivalent to feeding the
image to the lifter): we compare against the feed-forward baseline that lifts the occluded
image directly, and on Objaverse we verify with GT novel-view PSNR over the FULL object.

Two ways the observation carries only partial info (both supported):
  --occ right|left|lower|blob   : appearance of part of the object is hidden
  --pose                        : also recover camera rotation (generation != observed pose)
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
from ig import rotate_splats, dino_feats, rand_rot
from utils.abs_utils import symmetric_orthogonalization
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/ig2_out"


def make_visibility(M, occ, frac=0.5):
    """M [1,1,128,128] fg. Return V (observed fg) and O (occluded fg)."""
    m = M[0, 0].bool().cpu().numpy()
    ys, xs = np.where(m)
    O = np.zeros_like(m)
    if len(xs) and occ != "none":
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if occ == "right":
            O[:, int(x1 - frac * (x1 - x0)):] = True
        elif occ == "left":
            O[:, :int(x0 + frac * (x1 - x0))] = True
        elif occ == "lower":
            O[int(y1 - frac * (y1 - y0)):, :] = True
        elif occ == "blob":
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            r = int(frac * max(x1 - x0, y1 - y0) / 2)
            yy, xx = np.ogrid[:128, :128]
            O[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True
    O = torch.from_numpy(O & m).float().view(1, 1, 128, 128).to(P.DEV)
    V = (M - O).clamp(0, 1)
    return V, O


def invert_partial(I, M, V, gp, eqm, vae, cams, lp, cfg, use_dino=True, K=8, nrot=2,
                   steps1=60, steps2=100, keep=3, eta=0.0017, pose=True, warm=True,
                   z_init=None, R_init=None, n_prior=1, src_cam=None, classid=None):
    if src_cam is None:
        src_cam = (cams[0][0], cams[1][0], cams[2][0])
    P0 = K * nrot
    z = torch.randn(P0, 4, 32, 32, device=P.DEV) if z_init is None else z_init.clone()
    if z_init is None and warm:
        with torch.no_grad():
            # warm start WITHOUT pinning occluded pixels to white: fill the occluded region
            # with the horizontal mirror of the visible content (uses only observed pixels,
            # no ground-truth leak) so EqM starts from a plausible complete object and is free
            # to hallucinate the hidden region from there.
            O = (M - V).clamp(0, 1)
            filled = I * (1 - O) + torch.flip(I, dims=[3]) * O
            filled = filled * M + (1 - M)
            z[0] = P.vae_encode(vae, F.interpolate(filled, (256, 256), mode="bilinear"))[0]
    R = torch.eye(3, device=P.DEV).unsqueeze(0).repeat(P0, 1, 1)
    if R_init is not None:
        R = R_init.clone()
    elif pose:
        R[1:] = rand_rot(P0 - 1, P.DEV)
    y = torch.full((P0,), 1000 if classid is None else classid, device=P.DEV)
    Iexp, Mexp, Vexp = I.expand(P0, -1, -1, -1), M.expand(P0, -1, -1, -1), V.expand(P0, -1, -1, -1)
    Idino = dino_feats(I * V + (1 - V)) if use_dino else None

    def fwd(zc, Rc, idx):
        it = torch.tensor(idx, device=P.DEV)
        img = P.vae_decode(vae, zc[it])
        u = F.interpolate(img, (128, 128), mode="bilinear") * Mexp[it] + (1 - Mexp[it])  # full silhouette
        losses = []
        for j, k in enumerate(idx):
            sp = P.lift(gp, u[j:j+1]); spr = rotate_splats(sp, Rc[k]) if pose else sp
            r = P.render_view(cfg, spr, src_cam[0], src_cam[1], src_cam[2])
            L = ((Vexp[k] * (r - Iexp[k])) ** 2).sum() / (Vexp[k].sum() + 1) \
                + 0.3 * lp((r * Vexp[k]).unsqueeze(0) * 2 - 1, (Iexp[k] * Vexp[k]).unsqueeze(0) * 2 - 1).mean()
            if use_dino:
                L = L + 0.5 * (1 - (dino_feats((r * Vexp[k] + (1 - Vexp[k])).unsqueeze(0)) * Idino).sum(-1).mean())
            losses.append(L)
        return torch.stack(losses)

    alive = list(range(P0))
    for stage, ns in [(1, steps1), (2, steps2)]:
        zp = z.clone().detach().requires_grad_(True)
        Rp = R.clone().detach().requires_grad_(pose)
        opt = torch.optim.Adam([zp] + ([Rp] if pose else []), lr=0.04)
        for t in range(ns):
            opt.zero_grad(); fwd(zp, Rp, alive).sum().backward(); opt.step()
            with torch.no_grad():
                # project z back onto the EqM natural-image manifold (several equilibrium steps).
                # This is what forces the search to explain an unnatural observation via POSE
                # rather than by generating an off-manifold (e.g. top-down) image.
                for _ in range(n_prior):
                    zp.data[alive] += eta * P.eqm_grad(eqm, zp.detach()[alive], y[alive])
                if pose:
                    Rp.data[alive] = symmetric_orthogonalization(Rp.data[alive])
        z, R = zp.detach(), Rp.detach()
        with torch.no_grad():
            per = fwd(z, R, alive)
        order = [alive[i] for i in torch.argsort(per).tolist()]
        if stage == 1:
            alive = order[:keep]
    best = order[0]
    with torch.no_grad():
        img = P.vae_decode(vae, z[best:best+1]); recon = F.interpolate(img, (128, 128), mode="bilinear")[0]
        sp = P.lift(gp, (recon * M[0] + (1 - M[0])).unsqueeze(0))
        spr = rotate_splats(sp, R[best]) if pose else sp
    return spr, recon, R[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--occ", default="right", choices=["none", "right", "left", "lower", "blob"])
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tt", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae()
    cams = P.build_turntable(cfg, num=a.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()
    I, M, tg, gt = ig.load_obs(a.obs, cfg); tag = a.tag or f"{tg}_{a.occ}"
    V, O = make_visibility(M, a.occ, a.frac)
    print(f"obs {tag} occ={a.occ} visible fg={100*V.sum()/M.sum():.0f}%")

    # feed-forward baseline: lift the OCCLUDED image directly (occluded region -> white)
    occ_img = (I * V + (1 - V))
    with torch.no_grad():
        ff = P.lift(gp, occ_img)
    # inversion (prior completes the hidden region)
    spr, recon, R = invert_partial(I, M, V, gp, eqm, vae, cams, lp, cfg,
                                   use_dino=not a.no_dino, pose=not a.no_pose)
    # completion metric: how well the hallucinated occluded region matches the held-out truth
    Occ = (M - V).clamp(0, 1)
    if float(Occ.sum()) > 0:
        comp = float(-10 * torch.log10(((Occ[0] * (recon - I[0])) ** 2).sum() / (Occ.sum() * 3 + 1) + 1e-12))
        print(f"[{tag}] occluded-region completion PSNR (inversion vs held-out truth) {comp:.2f} dB")

    def novel(splats):
        if gt is None:
            return None
        ps = []
        for v in range(1, gt["wv"].shape[0]):
            with torch.no_grad():
                r = P.render_view(cfg, splats, gt["wv"][v], gt["fp"][v], gt["cc"][v])
            ps.append(float(-10 * torch.log10(((r - gt["imgs"][v]) ** 2).mean() + 1e-12)))
        return float(np.mean(ps))
    ff_p, inv_p = novel(ff), novel(spr)
    print(f"[{tag}] feed-forward(occluded) novel PSNR {ff_p} | inversion novel PSNR {inv_p}")

    ff_tt = P.turntable_frames(cfg, ff, cams)
    inv_tt = P.turntable_frames(cfg, spr, cams)
    disp_t = I[0] * V[0] + 0.5 * (M[0] - V[0]).clamp(0, 1) + (1 - M[0])          # occluded region -> gray
    disp = (disp_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rec = (recon.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([disp, rec, ff, iv], 1) for ff, iv in zip(ff_tt, inv_tt)]
    imageio.mimsave(f"{OUT}/p_{tag}.mp4", frames, fps=20, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/p_{tag}_strip.png")
    print(f"saved p_{tag}.mp4 (partial input | inverted recon | feed-forward 3D | inversion 3D)")


if __name__ == "__main__":
    main()
