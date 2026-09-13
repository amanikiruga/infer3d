"""OPTION 1 -- pose reasoning through search (Infer3D SO(3) with an EqM prior).

Observe an UNNATURAL (most top-down) Objaverse view. Feed-forward lifting of that view gives
bad 3D. Infer3D inverts: the EqM natural-image energy prior refuses a top-down generation
(off-manifold) and instead produces a NATURAL (side) view, while the pose search finds the
rotation R that maps it to the top-down observation. Verified on the held-out NATURAL views
(GT novel-view PSNR) + turntable, vs the feed-forward baseline.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
from ig2 import invert_partial
from eqm_sanity import nag_gd_sample


def Rx(deg):
    t = np.deg2rad(deg)
    return torch.tensor([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]],
                        dtype=torch.float32, device=P.DEV)


def Ry(deg):
    t = np.deg2rad(deg)
    return torch.tensor([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]],
                        dtype=torch.float32, device=P.DEV)


def Rz(deg):
    t = np.deg2rad(deg)
    return torch.tensor([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]],
                        dtype=torch.float32, device=P.DEV)

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/pose_out"


def novel(cfg, splats, gt):
    """Appearance-robust pose+shape verification: silhouette IoU of the recovered 3D vs GT
    across the held-out NATURAL views. (Pixel PSNR is unfair here since EqM produces a generic
    natural instance, not the exact object's texture.) Also returns mean PSNR for reference."""
    ious, ps = [], []
    for v in range(1, gt["wv"].shape[0]):
        with torch.no_grad():
            r = P.render_view(cfg, splats, gt["wv"][v], gt["fp"][v], gt["cc"][v])
        gtm = (gt["imgs"][v] < 0.985).any(0).float()
        rm = (r < 0.985).any(0).float()
        inter = (gtm * rm).sum(); union = (gtm + rm - gtm * rm).sum()
        ious.append(float(inter / (union + 1e-6)))
        ps.append(float(-10 * torch.log10(((r - gt["imgs"][v]) ** 2).mean() + 1e-12)))
    return float(np.mean(ious)), float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--classid", type=int, default=849, help="ImageNet class for natural hypotheses (teapot=849)")
    ap.add_argument("--K", type=int, default=4, help="natural latent hypotheses")
    ap.add_argument("--steps1", type=int, default=80)
    ap.add_argument("--steps2", type=int, default=120)
    ap.add_argument("--eta", type=float, default=0.002)
    ap.add_argument("--nprior", type=int, default=4)
    ap.add_argument("--tt", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae()
    cams = P.build_turntable(cfg, num=a.tt)
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()

    td, elevs = ig.objaverse_topdown_idx(cfg, a.uid)
    tag = a.tag or f"pose_{a.uid[:8]}"
    print(f"{tag}: top-down view idx={td} elev={elevs[td]:.0f}deg  all_elevs={[round(e) for e in elevs]}")

    it = ig.objaverse_item_uid(cfg, a.uid, src_idx=td)   # observation = top-down view
    I = it["gt_images"][0].unsqueeze(0).to(P.DEV)
    M = (I < 0.985).any(1, keepdim=True).float()
    gt = {"imgs": it["gt_images"].to(P.DEV), "wv": it["world_view_transforms"].to(P.DEV),
          "fp": it["full_proj_transforms"].to(P.DEV), "cc": it["camera_centers"].to(P.DEV)}

    with torch.no_grad():
        ff = P.lift(gp, I)                                # feed-forward on the top-down view
    ff_iou, ff_p = novel(cfg, ff, gt)

    # NATURAL hypotheses from the EqM prior (class-conditional) x a rotation set spanning
    # elevations/azimuths -> multi-start. The strong prior projection (n_prior EqM steps/iter)
    # keeps z on the natural manifold, so the search must explain the top-down view via POSE.
    with torch.no_grad():
        lat = nag_gd_sample(eqm, torch.full((a.K,), a.classid, device=P.DEV), 250, cfg=1.5)  # [K,4,32,32]
    ROTS = [Rx(45), Rx(70), Rx(-45), Rx(-70), Ry(55) @ Rx(60), Ry(-55) @ Rx(60)]
    nrot = len(ROTS)
    z_init = torch.stack([lat[i] for i in range(a.K) for _ in ROTS], 0)          # [K*nrot,4,32,32]
    R_init = torch.stack([R for _ in range(a.K) for R in ROTS], 0)               # [K*nrot,3,3]
    src_cam = (gt["wv"][0], gt["fp"][0], gt["cc"][0])   # fit at the true observation camera (consistent w/ GT eval)
    spr, recon, R = invert_partial(I, M, M.clone(), gp, eqm, vae, cams, lp, cfg, use_dino=True,
                                   K=a.K, nrot=nrot, steps1=a.steps1, steps2=a.steps2, eta=a.eta,
                                   pose=True, z_init=z_init, R_init=R_init, n_prior=a.nprior, keep=4,
                                   src_cam=src_cam)
    inv_iou, inv_p = novel(cfg, spr, gt)
    print(f"[{tag}] novel-view silhouette IoU  feed-forward {ff_iou:.3f} | Infer3D {inv_iou:.3f}   "
          f"(PSNR ff {ff_p:.1f} / inv {inv_p:.1f})")

    ff_tt = P.turntable_frames(cfg, ff, cams)
    inv_tt = P.turntable_frames(cfg, spr, cams)
    inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rec = (recon.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, rec, f, iv], 1) for f, iv in zip(ff_tt, inv_tt)]
    imageio.mimsave(f"{OUT}/{tag}.mp4", frames, fps=20, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/{tag}_strip.png")
    print(f"saved {tag}.mp4 (top-down input | recovered natural recon | feed-forward 3D | Infer3D 3D)")


if __name__ == "__main__":
    main()
